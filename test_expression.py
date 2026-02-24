import asyncio
from backend.pipeline.tissue_expression import TissueExpressionScorer
scorer = TissueExpressionScorer("lung cancer")
ok, msg = asyncio.run(scorer.validate_api_connection("EGFR"))
print(ok, msg)  # should print True + EGFR lung expression level