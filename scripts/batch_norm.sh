# script to normalise set of tubs in parallel

tubs='data/tub_4[6-9]*_*/ data/tub_[5-6]*/'
#tubs='data/tub_46_*/'

# here we use shell wildcard expansion
tub_list=`ls -d $tubs`
echo $tub_list

# enforces Ctrl-C to stop the asynchronous processes
trap "kill 0" EXIT

for tub in $tub_list; do
    donkey tubnorm "$tub" &
done

# required for the script to return
wait
exit 0

