#!/bin/sh

for fn in "Koala" "Penguins";
do
    for k in 2 5 10 15 20;
    do
        java KMeans.java "$fn".jpg "$k" "${fn}${k}.jpg"
    done
done
