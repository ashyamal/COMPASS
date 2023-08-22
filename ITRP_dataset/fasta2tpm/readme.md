## setup env


1. download micromamba, add to bashrc:
   ```bash
   export PATH=$PATH:~/Software/bin/
   eval "$(micromamba shell hook --shell bash)"
   ```
   
2. install required packages:
   ```bash
   micromamba create -n RNA
   micromamba activate RNA
   micromamba install -y -c hcc aspera-cli
   micromamba install -y -c bioconda fastqc multiqc fastp samtools star
   micromamba install pip jupyter jupyterlab
    ```