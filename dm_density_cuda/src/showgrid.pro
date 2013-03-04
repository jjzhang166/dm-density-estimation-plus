FUNCTION showgrid, filename
    head = lonarr(64);
    openr,lun,filename,/get_lun
    readu,lun,head
    gridsize = head(0) 
    print, gridsize
    den = fltarr(gridsize* gridsize* gridsize);
    readu,lun,den
    close,lun

    den = reform(den, gridsize, gridsize, gridsize)
    loadct, 0
    window, xs=512,ys=512
    tvscale, den[*,*, gridsize/2]^0.001,/nointerpolation
    return,den
END
