#!/bin/bash

FILE=$1

if [[ ${FILE} != "apple2orange" && ${FILE} != "summer2winter_yosemite" &&  ${FILE} != "horse2zebra" && ${FILE} != "monet2photo" && ${FILE} != "cezanne2photo" && ${FILE} != "ukiyoe2photo" && ${FILE} != "vangogh2photo" && ${FILE} != "maps" && ${FILE} != "facades" && ${FILE} != "iphone2dslr_flower" ]]; then
    echo "Available datasets are: apple2orange, summer2winter_yosemite, horse2zebra, monet2photo, cezanne2photo, ukiyoe2photo, vangogh2photo, maps, cityscapes, facades, iphone2dslr_flower"
    exit 1
fi

URL=https://people.eecs.berkeley.edu/~taesung_park/CycleGAN/datasets/${FILE}.zip
ZIP_FILE=${FILE}.zip
TARGET_DIR=${FILE}
wget ${URL}
unzip ${ZIP_FILE}
rm ${ZIP_FILE}
