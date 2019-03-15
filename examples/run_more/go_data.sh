#

# get the treebank
wget https://lindat.mff.cuni.cz/repository/xmlui/bitstream/handle/11234/1-2837/ud-treebanks-v2.2.tgz
tar -xzvf ud-treebanks-v2.2.tgz

# data directory and codes
git clone https://github.com/uclanlp/CrossLingualDepParser src
mkdir -p ./data2.2_more/
git clone https://github.com/Babylonpartners/fastText_multilingual data2.2_more/fastText_multilingual
python3 ./src/examples/run_more/prepare_data.py |& grep -v "s$" | tee data2.2_more/log

# now you have "*_{train,dev,test}.conllu" and "wiki.multi.*.vec" in the "data2.2_more" dir
