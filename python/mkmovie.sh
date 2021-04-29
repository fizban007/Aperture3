ffmpeg -y -f image2 -r 12 -i plots/%05d.png -c:v libx264 -crf 18 -pix_fmt yuv420p -vf "pad=ceil(iw/2)*2:ceil(ih/2)*2" 1d_gr_gap_KS_lic0.01.mp4
