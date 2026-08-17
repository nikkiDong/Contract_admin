#!/usr/bin/env python3
"""CDK app entry point."""

import os

import aws_cdk as cdk

from stack import ContractRiskStack

app = cdk.App()

ContractRiskStack(
    app,
    "ContractClauseRiskStack",
    env=cdk.Environment(
        account=os.environ.get("CDK_DEFAULT_ACCOUNT"),
        region=os.environ.get("CDK_DEFAULT_REGION"),
    ),
    description="Contract Clause Risk Flagging - evidence-grounded review service.",
)

app.synth()
