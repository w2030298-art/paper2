# Paper2 N1 Oracle Gap

The v4.9 matrix runner marks `N1-oracle-small` records as `benchmark_diagnostic` because no local oracle wrapper is wired into this patch. These records may support diagnostic benchmark checks, but they must not be claimed as oracle-gap evidence until an oracle-capable runner is added and validated.
