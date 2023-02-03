call D:/anaconda3/Scripts/activate.bat D:/anaconda3/
call conda activate general_ds
call cd /d D:\LEARN\DATA SCIENCE\2 WORKSPACE\4 PROJECT\0 BOOK\mybookcollection\jabodetabekhousepricing
call rmdir /s /q "_build"
call cd /d D:\LEARN\DATA SCIENCE\2 WORKSPACE\4 PROJECT\2 JABODETABEK HOUSE PRICING\jabodetabekhousepricing
call cd ..
call jupyter-book build --path-output "D:\LEARN\DATA SCIENCE\2 WORKSPACE\4 PROJECT\0 BOOK\mybookcollection\jabodetabekhousepricing" jabodetabekhousepricing
