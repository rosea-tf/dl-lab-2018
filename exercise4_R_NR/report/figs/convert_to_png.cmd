for %%f in (*.svg) do (
    rsvg-convert -w 960 "%%~nf.svg" > "%%~nf.png" 
)