{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 3,
   "metadata": {},
   "outputs": [],
   "source": [
    "import numpy as np\n",
    "import pandas as pd\n",
    "from pathlib import Path\n",
    "from scipy.io import loadmat\n",
    "import matplotlib.pyplot as plt"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "metadata": {},
   "outputs": [
    {
     "ename": "IndentationError",
     "evalue": "unindent does not match any outer indentation level (<tokenize>, line 3)",
     "output_type": "error",
     "traceback": [
      "\u001b[1;36m  File \u001b[1;32m<tokenize>:3\u001b[1;36m\u001b[0m\n\u001b[1;33m    data = loadmat(raw_data)\u001b[0m\n\u001b[1;37m    ^\u001b[0m\n\u001b[1;31mIndentationError\u001b[0m\u001b[1;31m:\u001b[0m unindent does not match any outer indentation level\n"
     ]
    }
   ],
   "source": [
    "def data_cleaning(raw_data):\n",
    "    data = loadmat(raw_data)\n",
    "    for key, value in data.items():\n",
    "        if key.endswith('DE_time'):\n",
    "            df = pd.DataFrame(value, columns=[key])\n",
    "            signal = df.to_numpy()\n",
    "            signal = signal.flatten()\n",
    "    x = np.count_nonzero(np.isnan(signal))\n",
    "    if x >= 1:\n",
    "        raise ValueError(\"nan is found in data.\")\n",
    "    return signal"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "tf",
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
   "version": "3.10.12"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}
