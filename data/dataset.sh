#mkdir cityscape && cd $_
#wget "https://www.cityscapes-dataset.com/file-handling/?packageID=1"
#filename=$(awk -F'/' {print $NF})
#unzip "$filename"
#rm "$filename"

#cd -
mkdir gta5 && cd $_
wget "https://download.visinf.tu-darmstadt.de/data/from_games/data/01_images.zip"
filename=$(awk -F'/' {print $NF})
unzip "$filename"
rm "$filename"
wget "https://download.visinf.tu-darmstadt.de/data/from_games/data/01_labels.zip"
filename=$(awk -F'/' {print $NF})
unzip "$filename"
rm "$filename"
