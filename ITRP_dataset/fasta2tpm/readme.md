## setup env


1. download micromamba, add to bashrc:
   ```bash
   mkdir -p ~/Software/
   cd ~/Software/
   curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest | tar -xvj bin/micromamba
   export PATH=$PATH:~/Software/bin/
   eval "$(micromamba shell hook --shell bash)"
   ```
   
3. install required packages:
   ```bash
   micromamba create -n RNA
   micromamba activate RNA
   micromamba install -y -c hcc aspera-cli
   micromamba install -y -c bioconda fastqc multiqc fastp samtools star
   micromamba install pip jupyter jupyterlab
   pip install rnanorm
    ```