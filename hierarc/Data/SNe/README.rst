# Pantheon data set

Source: https://github.com/dscolnic/Pantheon

The binned data file can be found here:
https://github.com/dscolnic/Pantheon/blob/master/Binned_data/lcparam_DS17f.txt

# Pantheon+SH0ES data set

Source: https://github.com/PantheonPlusSH0ES/DataRelease/tree/main/Pantheon%2B_Data/4_DISTANCES_AND_COVAR

Used through ``sample_name='PantheonPlus'``.

# Original DES-SN5YR data set

Source: https://github.com/des-science/DES-SN5YR/tree/main/4_DISTANCES_COVMAT (now outdated, see below)

``DES-SN5YR/DES-SN5YR_HD.csv`` and ``DES-SN5YR/STAT+SYS.txt``.
Used through ``sample_name='DES5YR'``.

# DES-SN5YR-Dovekie data set

Source: https://github.com/des-science/DES-SN5YR/tree/main/4_DISTANCES_COVMAT

``DES-SN5YR-Dovekie/DES-Dovekie_HD.csv`` and ``DES-SN5YR-Dovekie/STAT+SYS.npz``, the DES-SN5YR
sample re-calibrated with Dovekie. Used through ``sample_name='DES5YR_Dovekie'``.

Note the two format changes relative to the original DES-SN5YR release, which are handled in
``hierarc/Likelihood/SneLikelihood/des_5yr_dovekie.py`` following the released
``DES-Dovekie-SN_Likelihood.py``:

* ``STAT+SYS.npz`` stores the **inverse** stat+sys covariance matrix (1820x1820), in
  upper-triangular packed form. It is unpacked, symmetrized and inverted back to a covariance
  matrix. Since the stored matrix already includes the statistical errors, ``MUERR`` must **not**
  be added to the diagonal again.
* the Hubble diagram column with the distance modulus uncertainty is ``MUERR``, not
  ``MUERR_FINAL``, and the file is white-space separated in the SNANA style
  (``VARNAMES:`` header, ``SN:`` row keys) despite the ``.csv`` extension.

``DES-Dovekie_HD.csv`` is ordered consistently with the covariance matrix; the metadata file of the
release has a different ordering and must not be used together with the covariance matrix.
