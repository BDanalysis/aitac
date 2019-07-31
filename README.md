# aitac
1. Installation:

1.1 Basic requirements:

Software: Python, SAMtools
Operating System: Linux, Windows
Memory: At least 4 GB of memory
Python version: 3
Numpy, Scipy, Pandas，pysam

1.2 Download:

Download the compressed file iftv.tar.GZ and then do the following:

$ Tar-xzvf iftv.tar.gz

After decompression, two files, iftv.py and run.py, as well as two files calc.py and calculate1.py, which are convenient for statistical average after batch running samples.

2. Running software:

2.1 Preprocessing of input files:

Usually, the following documents are required:

A BAM file from a tumor sample.
A genome reference sequence FASTA file.

If your sample starts from the fastq file, you can do the following:

$BWA MEM ref.fa example.fq > example.sam
$Satools view - bS example. Sam > example. BAM
$samtools sort example.bam example.sort

2.2 Operating method:

First, modify run.py:

ChrFile: The path to the FASTA file of the genome reference sequence used by the user.
RdFile: A BAM file representing the sample used by the user.
OutputFile: The results of the copy number detected by CNV-IFTV and the absolute copy number restored by the predicted tumor purity.
StatisticFile: Contains predicted tumor purity.
CalulateFile: The average absolute copy number of all samples counted after running the samples in batches.
BeforeFile: After running the samples in batches, the average copy number detected before the tumor purity was restored.

Then the following operations are performed in the software catalog:

$Python run.py

If you need to count the average copy number after batch running, do the following:

$Python calculate.py
$Python calculate1.py

