CONDA_ENV="atlas"
ENCRYPTED_DATA_URL="ftp://www.nitrc.org/fcon_1000/htdocs/indi/retro/ATLAS/releases/R1.1/ATLAS_R1.1_encrypted.tar.gz"
ENCRYPTED_FILENAME="ATLAS_R1.1_encrypted.tar.gz"
DECRYPTED_FILENAME="ATLAS_R1.1_decrypted.tar.gz"


HEAD_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
CODE_DIR=$HEAD_DIR/code
DATA_DIR=$HEAD_DIR/data
EXP_DIR=$HEAD_DIR/experiments

mkdir -p $EXP_DIR

# Creates the environment
conda create -n $CONDA_ENV python=3.6

# Activates the environment
source activate $CONDA_ENV

# pip install into environment
pip install -r requirements.txt

# Downloads and preprocesses ATLAS data and saves in data/
mkdir -p "$DATA_DIR"

if [ ! -f "$DATA_DIR/$ENCRYPTED_FILENAME" ]; then
  echo "Downloading data from $ENCRYPTED_DATA_URL to $DATA_DIR/$ENCRYPTED_FILENAME"
  curl $ENCRYPTED_DATA_URL > "$DATA_DIR/$ENCRYPTED_FILENAME"
else
  echo "Skipping downloading data from $ENCRYPTED_DATA_URL; $DATA_DIR/$ENCRYPTED_FILENAME already exists!"
fi

if [ ! -f "$DATA_DIR/$DECRYPTED_FILENAME" ]; then
  echo "Decrypting data from $DATA_DIR/$ENCRYPTED_FILENAME to $DATA_DIR/$DECRYPTED_FILENAME"
  openssl aes-256-cbc -d -a -in "$DATA_DIR/$ENCRYPTED_FILENAME" -out "$DATA_DIR/$DECRYPTED_FILENAME"
else
  echo "Skipping Decrypting data from $DATA_DIR/$ENCRYPTED_FILENAME to $DATA_DIR/$DECRYPTED_FILENAME; $DATA_DIR/$DECRYPTED_FILENAME already exists!"
fi

if [ ! -d "$DATA_DIR/ATLAS_R1.1" ]; then
  echo "Decompressing data from $DATA_DIR/$DECRYPTED_FILENAME to $DATA_DIR"
  tar -xzf "$DATA_DIR/$DECRYPTED_FILENAME" -C $DATA_DIR
else
  echo "Skipping decompressing data from $DATA_DIR/$DECRYPTED_FILENAME to $DATA_DIR; $DATA_DIR/ATLAS_R1.1 already exists!"
fi

function unpack_niigz()
{
  # local unpackDir=$(dirname $1)
  local filename=$1
  unpackDir="${filename%.nii.gz}"
  echo "Unpacking inputFile $filename into outputDir $unpackDir"
  med2image --inputFile $filename --outputDir $unpackDir --outputFileStem image --outputFileType jpg --sliceToConvert -1
}
export -f unpack_niigz

# Runs unpack_niigz for every file that matches *.nii.gz in $DATA_DIR
find $DATA_DIR -name "*.nii.gz" -type f -exec bash -c 'unpack_niigz {}' \;

# find . -type f -wholename "*Site*/*/*/*_t1w_deface_stx/*.jpg" | wc -l
