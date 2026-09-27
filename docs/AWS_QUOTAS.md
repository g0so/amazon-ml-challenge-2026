# AWS quota diagnosis

Verified against official AWS documentation on 2026-09-25. Your AWS account was
not accessed or changed. Region and account plan details remain unverified.

## Your exact blocker

Your error reports quota 0, usage 0, requested increment 1 for:
**Studio JupyterLab Apps running on ml.g6.xlarge instances**.

This is a launch quota for that instance and application type, not a Python memory
error and not proof that all 16 GB CPU instances are unavailable. AWS's published
SageMaker quota table lists a default of 0 for this GPU app type. G6 is a GPU
instance family; a GPU is unnecessary for our current phase.

- [SageMaker quota table](https://docs.aws.amazon.com/general/latest/gr/sagemaker.html)
- [G6 instance family](https://aws.amazon.com/ec2/instance-types/g6/)

## Optional diagnostic steps (do not delay local work)

1. Confirm the Region in which the failed JupyterLab space was created.
2. In that same Region, open Service Quotas → AWS services → Amazon SageMaker AI.
3. Find the EXACT quota name above. Notebook-instance, training-job, endpoint,
   and Studio JupyterLab quotas are distinct; increasing the wrong one won't help.
4. If available and adjustable, request an applied quota of 1. Approval can be
   denied or delayed; requesting a quota does not itself launch the instance.
5. For CPU-only exploration, inspect the applied JupyterLab quota for a CPU
   instance such as ml.m5.xlarge. Do not infer access from the published default;
   actual account values, permissions, availability, and plan restrictions matter.
6. Record the exact response if the request is blocked. Do not upgrade the account
   merely to troubleshoot: paid-plan billing exposure is a separate decision.

[AWS request procedure](https://docs.aws.amazon.com/servicequotas/latest/userguide/request-quota-increase.html)
explains adjustability and approval. [AWS supported regions and quotas](https://docs.aws.amazon.com/sagemaker/latest/dg/regions-quotas.html)
explains where to inspect actual quotas.

## What the 4 GB and credits mean

AWS lists ml.t3.medium as a 4 GiB instance. That is the memory of a particular
instance, not a universal 4 GB free-account ceiling. Your running instance type
still needs confirmation. Credits cover eligible charges; they do not establish
quota or instance access.

[Instance specifications](https://docs.aws.amazon.com/sagemaker/latest/dg/notebooks-available-instance-types.html)
are documented for Studio Classic; actual availability in current Studio must be
checked in your Region and application.

AWS distinguishes Free and Paid account plans. The Free plan has restricted
service/feature access. Upgrading may preserve remaining applicable credits, but
can expose you to charges beyond credits and does not guarantee quota approval.
The supplied error alone does not establish that your plan caused the zero quota.

[AWS account-plan documentation](https://docs.aws.amazon.com/awsaccountbilling/latest/aboutv2/free-tier-plans.html)

Do not confuse Studio with Studio Lab. Studio Lab documentation describes 16 GB
RAM, but currently says it is closed to new customers, so it is not a dependable
new signup alternative: [Studio Lab overview](https://docs.aws.amazon.com/sagemaker/latest/dg/studio-lab-overview.html).

Decision: use Ubuntu locally now; keep AWS credits for a later, measured need.
No cloud upgrade, quota request, resource launch, or cloud charge was made here.
