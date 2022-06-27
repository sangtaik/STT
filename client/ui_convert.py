{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "id": "c2ebc3e1",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "converting ui_main.ui\n"
     ]
    }
   ],
   "source": [
    "from PyQt5 import uic\n",
    "import glob\n",
    "for fname in glob.glob(\"*.ui\"):\n",
    "    print(\"converting\",fname)\n",
    "    fin = open(fname,'r')\n",
    "    fout = open(fname.replace(\".ui\",\".py\"),'w')\n",
    "    uic.compileUi(fin,fout,execute=False)\n",
    "    fin.close()\n",
    "    fout.close()"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.9.7"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
