for i in 1 2 3 4 5; do
    python src/play.py -e $1 -d src/logs/evolution.py-c=onepoint,clp=0.3,cp=0.1,cwp=0.01,m=normal,mlp=0.3,mp=0.3,ms=0.07,mwp=0.01
done
