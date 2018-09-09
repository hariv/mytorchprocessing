for image in ~/fugazi/defense/data/deep_disc/*_fake_B.png; do
    #echo $image
    python resize.py "${image}"
done
