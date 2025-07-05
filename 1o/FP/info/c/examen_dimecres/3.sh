#!/bin/bash

echo "Escriu un nom de directory"

read nomdir

if [ -d "$nomdir" ] then
    echo "El directori existeix"
else
    mkdir $nomdir
    echo "He creat el directori"
fi