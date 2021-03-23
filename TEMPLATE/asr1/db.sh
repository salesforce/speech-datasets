# We extract WSJ0_TGZ to WSJ0 and WSJ1_TGZ to WSJ1. Note that the actual data
# is in WSJ0/csr_1_senn and WSJ1/csr_senn
WSJ0_TGZ=/export/data/LDC/csr_1_senn_LDC93S6B.tgz
WSJ1_TGZ=/export/data/LDC/csr_senn_LDC94S13B.tgz
WSJ0=/workspace/LDC93S6B
WSJ1=/workspace/LDC94S13B

# Extract SWBD1_TGZ to SWBD1
SWBD1_TGZ=/export/data/LDC/swb1_LDC97S62.tgz
SWBD1=/workspace/LDC97S62

# Filepath i of EVAL2000_TGZ extracts into directory i of EVAL2000.
# First directory must contain the speech data, second directory must contain the transcripts.
EVAL2000_TGZ="/export/data/LDC/hub5e_00_LDC2002S09.tgz /export/data/LDC/LDC2002T43.tgz"
EVAL2000="/workspace/LDC2002S09/hub5e_00 /workspace/LDC2002T43"

# Extract RT03_TGZ to RT03
RT03_TGZ=/export/data/LDC/rt_03_LDC2007S10.tgz
RT03=/workspace/LDC2007S10/rt_03

# filepath i of FISHER_TGZ extracts into directory i of FISHER
# In this case, we extract LDC2004T19 and LDC2005T19 every time, but LDC2004S13 and LDC2005S13 are pre-extracted
FISHER="/workspace/LDC2004T19 /workspace/LDC2005T19 /export/data/LDC/LDC2004S13 /export/data/LDC/LDC2005S13"
FISHER_TGZ="/export/data/LDC/LDC2004T19/fe_03_p1_tran_LDC2004T19.tgz /export/data/LDC/LDC2005T19/LDC2005T19.tgz"

LIBRISPEECH=/export/data/librispeech

COMMONVOICE=/export/data/commonvoice
