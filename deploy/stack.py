"""CDK stack for the contract clause risk flagging service.

    S3 (packages + reference)  ->  Lambda (pipeline)  ->  DynamoDB (findings)
                                        ^
                                        |
                            API Gateway (API key required)

Security posture, stated explicitly because it is easy to get wrong:

* The REST API **requires an API key** on /analyze and /findings. An unauthenticated
  endpoint would expose contract review findings to anyone with the URL. An API key
  is appropriate for a hackathon demo; it is a shared secret, not per-user identity.
  For anything beyond a demo, replace it with a Cognito user pool authorizer or IAM
  auth - see the note in SOLUTION.md. /health is intentionally open so a smoke test
  needs no secret, and it returns no contract data.
* The S3 bucket blocks all public access, enforces SSL, and is encrypted.
* The Lambda role is scoped to the specific bucket prefix and the single table.
  It is granted no Bedrock permission by default, because the deterministic path
  needs none.
* Removal policies are DESTROY so `cdk destroy` leaves nothing behind. That is
  correct for an ephemeral workshop account and wrong for production.
"""

from __future__ import annotations

from pathlib import Path

from aws_cdk import (
    CfnOutput,
    Duration,
    RemovalPolicy,
    Stack,
)
from aws_cdk import aws_apigateway as apigw
from aws_cdk import aws_dynamodb as dynamodb
from aws_cdk import aws_lambda as lambda_
from aws_cdk import aws_logs as logs
from aws_cdk import aws_s3 as s3
from constructs import Construct

BUNDLE = Path(__file__).resolve().parent.parent / "build" / "lambda_bundle"
PACKAGE_PREFIX = "packages"
CHECKLIST_KEY = "reference/Reference_Checklist.csv"


class ContractRiskStack(Stack):
    def __init__(self, scope: Construct, construct_id: str, **kwargs) -> None:
        super().__init__(scope, construct_id, **kwargs)

        if not BUNDLE.exists():
            raise FileNotFoundError(
                f"Lambda bundle not found at {BUNDLE}. Run: python deploy/build.py"
            )

        # -- storage ----------------------------------------------------
        packages = s3.Bucket(
            self,
            "PackagesBucket",
            block_public_access=s3.BlockPublicAccess.BLOCK_ALL,
            encryption=s3.BucketEncryption.S3_MANAGED,
            enforce_ssl=True,
            versioned=False,
            removal_policy=RemovalPolicy.DESTROY,
            auto_delete_objects=True,
        )

        findings = dynamodb.Table(
            self,
            "FindingsTable",
            partition_key=dynamodb.Attribute(
                name="document_id", type=dynamodb.AttributeType.STRING
            ),
            sort_key=dynamodb.Attribute(
                name="requirement_id", type=dynamodb.AttributeType.STRING
            ),
            billing_mode=dynamodb.BillingMode.PAY_PER_REQUEST,
            point_in_time_recovery_specification=(
                dynamodb.PointInTimeRecoverySpecification(
                    point_in_time_recovery_enabled=False
                )
            ),
            removal_policy=RemovalPolicy.DESTROY,
        )

        # -- compute ----------------------------------------------------
        log_group = logs.LogGroup(
            self,
            "AnalyserLogGroup",
            retention=logs.RetentionDays.ONE_WEEK,
            removal_policy=RemovalPolicy.DESTROY,
        )

        analyser = lambda_.Function(
            self,
            "AnalyserFunction",
            runtime=lambda_.Runtime.PYTHON_3_12,
            architecture=lambda_.Architecture.ARM_64,
            handler="handler.handler",
            code=lambda_.Code.from_asset(str(BUNDLE)),
            timeout=Duration.seconds(60),
            memory_size=512,
            log_group=log_group,
            environment={
                "PACKAGES_BUCKET": packages.bucket_name,
                "FINDINGS_TABLE": findings.table_name,
                "PACKAGE_PREFIX": PACKAGE_PREFIX,
                "CHECKLIST_KEY": CHECKLIST_KEY,
                "CRF_PDF_BACKEND": "pypdf",
                # Deterministic pipeline only. Set to "bedrock" and grant
                # bedrock:InvokeModel to enable residual adjudication.
                "LLM_PROVIDER": "null",
            },
        )

        # Least privilege: read only the two prefixes the handler uses.
        packages.grant_read(analyser, f"{PACKAGE_PREFIX}/*")
        packages.grant_read(analyser, CHECKLIST_KEY)
        findings.grant_read_write_data(analyser)

        # -- api --------------------------------------------------------
        api = apigw.RestApi(
            self,
            "ContractRiskApi",
            rest_api_name="contract-clause-risk-flagging",
            description="Evidence-grounded contract clause risk flagging (demo).",
            deploy_options=apigw.StageOptions(
                stage_name="prod",
                throttling_rate_limit=20,
                throttling_burst_limit=10,
                logging_level=apigw.MethodLoggingLevel.INFO,
                metrics_enabled=True,
            ),
            cloud_watch_role=True,
        )

        integration = apigw.LambdaIntegration(analyser)

        # /health - open on purpose: no contract data, used for smoke tests.
        api.root.add_resource("health").add_method(
            "GET", integration, api_key_required=False
        )

        # /analyze and /findings return contract review content: key required.
        api.root.add_resource("analyze").add_method(
            "POST", integration, api_key_required=True
        )
        api.root.add_resource("findings").add_resource("{document_id}").add_method(
            "GET", integration, api_key_required=True
        )

        key = api.add_api_key("DemoApiKey")
        api.add_usage_plan(
            "DemoUsagePlan",
            api_stages=[apigw.UsagePlanPerApiStage(api=api, stage=api.deployment_stage)],
            throttle=apigw.ThrottleSettings(rate_limit=20, burst_limit=10),
        ).add_api_key(key)

        # -- outputs ----------------------------------------------------
        CfnOutput(self, "ApiUrl", value=api.url)
        CfnOutput(self, "PackagesBucketName", value=packages.bucket_name)
        CfnOutput(self, "FindingsTableName", value=findings.table_name)
        CfnOutput(self, "ApiKeyId", value=key.key_id,
                  description="Retrieve with: aws apigateway get-api-key "
                              "--api-key <id> --include-value")
