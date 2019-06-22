#!/bin/bash
. startvenv.sh
for m in normal uniform;  do
  for c in onepoint twopoint; do
    for cp in 0.1 0.2 0.3; do
      for mp in 0.1 0.2 0.3; do
        for ms in 0.01 0.03 0.05 0.07 0.09; do
	  echo "python ./src/evolution.py -m $m -c $c -cp $cp -mp $mp -ms $ms"
          python ./src/evolution.py -m $m -c $c -cp $cp -mp $mp -ms $ms &
	done
        wait
      done
    done
  done
done
 
	
