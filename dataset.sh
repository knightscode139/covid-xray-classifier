#!/usr/bin/bash
unzip -q archive.zip
mkdir -p dataset/{covid,lung-opacity,normal,viral-pneumonia}
cd COVID-19_Radiography_Dataset
mv C*/i*/* ../dataset/covid
mv L*/i*/* ../dataset/lung-opacity
mv N*/i*/* ../dataset/normal
mv V*/i*/* ../dataset/viral-pneumonia
cd -
rm -rf COVID-19_Radiography_Dataset
