
```
for lmseed in 6 7 ; do
   for ftseed in 0 1 2 3; do
       python -m ulmfit poleval19_init data/reddit/pl-100/models/sp25k/lstm_nl4_lmseed-${lmseed}.m  dp1 --ftseed=${ftseed} --drop_mul_lm=1.0
   done
done

for lmseed in 4 5 ; do                                                         
   for ftseed in 0 1 2 3; do
       python -m ulmfit poleval19_init data/reddit/pl-100/models/sp25k/lstm_nl4_lmseed-${lmseed}.m  dp1 --ftseed=${ftseed} --drop_mul_lm=1.0
   done
done
   
   
   
for lmseed in 6 7 ; do                                                         
   python -m ulmfit poleval19_seeds "data/hate/pl-10-reddit/models/sp25k/lstm_dp1_lmseed-${lmseed}-*.m" --seed_name='clsweightseed'
done

for lmseed in 4 5 ; do                                                         
   python -m ulmfit poleval19_seeds "data/hate/pl-10-reddit/models/sp25k/lstm_dp1_lmseed-${lmseed}-*.m" --seed_name='clsweightseed'
done

   
```