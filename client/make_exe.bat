pyinstaller --noconfirm --clean --distpath ./dist ^
			--add-data="symspell_jamo_dict.txt;." --add-data="Assets/vocab.json;Assets" --add-data="Assets/vocab_jamos.json;Assets"^
			--add-data="Assets/test_data.wav;Assets" --add-data="Assets/vocab_chars.json;Assets"^
			--copy-metadata=tqdm ^
			--copy-metadata=regex --copy-metadata=requests ^
			--copy-metadata=packaging --copy-metadata=filelock --copy-metadata=numpy ^
			--copy-metadata=tokenizers --copy-metadata=importlib_metadata ^
			--collect-data=librosa --copy-metadata=librosa ^
			--hidden-import="sklearn.utils._cython_blas" ^
			--hidden-import="sklearn.utils._typedefs" ^
			--hidden-import="sklearn.neighbors._partition_nodes" ^
			--hidden-import="scipy.special.cython_special" ^
			main_run.py

dist\\main_run\\main_run.exe