

# Procedures
- change all absolute path in the submission script and input.yaml
- if you want a energy-only search, follow the comments in input.yaml
- submit the job by sbatch


# For other systems
- First you need to fit a set of instrument parameters manually in GSASII GUI.
You can follow this tutorial:
https://subversion.xray.aps.anl.gov/pyGSAS/Tutorials/CWInstDemo/FindProfParamCW.htm

- After that, replece the reference powder diffraction data (diamond.txt) and the instrument parameter (diamond.instprm) with the ones of your new systam.


# Resources

- GSASIIscriptable tutorial and API documentation:
https://gsas-ii.readthedocs.io/en/latest/GSASIIscriptable.html

- GSASII GUI tutorial:
https://subversion.xray.aps.anl.gov/pyGSAS/trunk/help/Tutorials.html

