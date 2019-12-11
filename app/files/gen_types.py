import glob
import os


s = ''
template = '{0} chembl_test_data/receptors/20014_receptor.gninatypes {1}\n'
for fname in glob.iglob('chembl_test_data/ligands/20014_*/*.gninatypes'):
    s += template.format(int(fname.find('active') != -1), fname)

with open('test_types.types', 'w') as f:
    f.write(s[:-1])
