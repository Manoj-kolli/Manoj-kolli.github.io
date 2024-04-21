In this post I'm going to run through a function in Python that can quickly find a Rock or Mine for below given input values.  For example, if I passed the function a values of sonar data below, it would find whether it is a Rock or Mine!

Let's get into it!

Importing the Dependencies


```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

```

Data Collection and Data Processing


```python
data = pd.read_csv(r"/content/drive/MyDrive/datasets/Copy of sonar data.csv", header= None)
```


```python
data.head()
```





  <div id="df-b2405124-3618-41ce-8a77-858d6f8c4088" class="colab-df-container">
    <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
      <th>4</th>
      <th>5</th>
      <th>6</th>
      <th>7</th>
      <th>8</th>
      <th>9</th>
      <th>...</th>
      <th>51</th>
      <th>52</th>
      <th>53</th>
      <th>54</th>
      <th>55</th>
      <th>56</th>
      <th>57</th>
      <th>58</th>
      <th>59</th>
      <th>60</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.0200</td>
      <td>0.0371</td>
      <td>0.0428</td>
      <td>0.0207</td>
      <td>0.0954</td>
      <td>0.0986</td>
      <td>0.1539</td>
      <td>0.1601</td>
      <td>0.3109</td>
      <td>0.2111</td>
      <td>...</td>
      <td>0.0027</td>
      <td>0.0065</td>
      <td>0.0159</td>
      <td>0.0072</td>
      <td>0.0167</td>
      <td>0.0180</td>
      <td>0.0084</td>
      <td>0.0090</td>
      <td>0.0032</td>
      <td>R</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.0453</td>
      <td>0.0523</td>
      <td>0.0843</td>
      <td>0.0689</td>
      <td>0.1183</td>
      <td>0.2583</td>
      <td>0.2156</td>
      <td>0.3481</td>
      <td>0.3337</td>
      <td>0.2872</td>
      <td>...</td>
      <td>0.0084</td>
      <td>0.0089</td>
      <td>0.0048</td>
      <td>0.0094</td>
      <td>0.0191</td>
      <td>0.0140</td>
      <td>0.0049</td>
      <td>0.0052</td>
      <td>0.0044</td>
      <td>R</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.0262</td>
      <td>0.0582</td>
      <td>0.1099</td>
      <td>0.1083</td>
      <td>0.0974</td>
      <td>0.2280</td>
      <td>0.2431</td>
      <td>0.3771</td>
      <td>0.5598</td>
      <td>0.6194</td>
      <td>...</td>
      <td>0.0232</td>
      <td>0.0166</td>
      <td>0.0095</td>
      <td>0.0180</td>
      <td>0.0244</td>
      <td>0.0316</td>
      <td>0.0164</td>
      <td>0.0095</td>
      <td>0.0078</td>
      <td>R</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.0100</td>
      <td>0.0171</td>
      <td>0.0623</td>
      <td>0.0205</td>
      <td>0.0205</td>
      <td>0.0368</td>
      <td>0.1098</td>
      <td>0.1276</td>
      <td>0.0598</td>
      <td>0.1264</td>
      <td>...</td>
      <td>0.0121</td>
      <td>0.0036</td>
      <td>0.0150</td>
      <td>0.0085</td>
      <td>0.0073</td>
      <td>0.0050</td>
      <td>0.0044</td>
      <td>0.0040</td>
      <td>0.0117</td>
      <td>R</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.0762</td>
      <td>0.0666</td>
      <td>0.0481</td>
      <td>0.0394</td>
      <td>0.0590</td>
      <td>0.0649</td>
      <td>0.1209</td>
      <td>0.2467</td>
      <td>0.3564</td>
      <td>0.4459</td>
      <td>...</td>
      <td>0.0031</td>
      <td>0.0054</td>
      <td>0.0105</td>
      <td>0.0110</td>
      <td>0.0015</td>
      <td>0.0072</td>
      <td>0.0048</td>
      <td>0.0107</td>
      <td>0.0094</td>
      <td>R</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 61 columns</p>
</div>
    <div class="colab-df-buttons">

  <div class="colab-df-container">
    <button class="colab-df-convert" onclick="convertToInteractive('df-b2405124-3618-41ce-8a77-858d6f8c4088')"
            title="Convert this dataframe to an interactive table."
            style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px" viewBox="0 -960 960 960">
    <path d="M120-120v-720h720v720H120Zm60-500h600v-160H180v160Zm220 220h160v-160H400v160Zm0 220h160v-160H400v160ZM180-400h160v-160H180v160Zm440 0h160v-160H620v160ZM180-180h160v-160H180v160Zm440 0h160v-160H620v160Z"/>
  </svg>
    </button>

  <style>
    .colab-df-container {
      display:flex;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    .colab-df-buttons div {
      margin-bottom: 4px;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

    <script>
      const buttonEl =
        document.querySelector('#df-b2405124-3618-41ce-8a77-858d6f8c4088 button.colab-df-convert');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      async function convertToInteractive(key) {
        const element = document.querySelector('#df-b2405124-3618-41ce-8a77-858d6f8c4088');
        const dataTable =
          await google.colab.kernel.invokeFunction('convertToInteractive',
                                                    [key], {});
        if (!dataTable) return;

        const docLinkHtml = 'Like what you see? Visit the ' +
          '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
          + ' to learn more about interactive tables.';
        element.innerHTML = '';
        dataTable['output_type'] = 'display_data';
        await google.colab.output.renderOutput(dataTable, element);
        const docLink = document.createElement('div');
        docLink.innerHTML = docLinkHtml;
        element.appendChild(docLink);
      }
    </script>
  </div>


<div id="df-b1505a36-905d-4651-bcc1-027ae9d6f01a">
  <button class="colab-df-quickchart" onclick="quickchart('df-b1505a36-905d-4651-bcc1-027ae9d6f01a')"
            title="Suggest charts"
            style="display:none;">

<svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
     width="24px">
    <g>
        <path d="M19 3H5c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h14c1.1 0 2-.9 2-2V5c0-1.1-.9-2-2-2zM9 17H7v-7h2v7zm4 0h-2V7h2v10zm4 0h-2v-4h2v4z"/>
    </g>
</svg>
  </button>

<style>
  .colab-df-quickchart {
      --bg-color: #E8F0FE;
      --fill-color: #1967D2;
      --hover-bg-color: #E2EBFA;
      --hover-fill-color: #174EA6;
      --disabled-fill-color: #AAA;
      --disabled-bg-color: #DDD;
  }

  [theme=dark] .colab-df-quickchart {
      --bg-color: #3B4455;
      --fill-color: #D2E3FC;
      --hover-bg-color: #434B5C;
      --hover-fill-color: #FFFFFF;
      --disabled-bg-color: #3B4455;
      --disabled-fill-color: #666;
  }

  .colab-df-quickchart {
    background-color: var(--bg-color);
    border: none;
    border-radius: 50%;
    cursor: pointer;
    display: none;
    fill: var(--fill-color);
    height: 32px;
    padding: 0;
    width: 32px;
  }

  .colab-df-quickchart:hover {
    background-color: var(--hover-bg-color);
    box-shadow: 0 1px 2px rgba(60, 64, 67, 0.3), 0 1px 3px 1px rgba(60, 64, 67, 0.15);
    fill: var(--button-hover-fill-color);
  }

  .colab-df-quickchart-complete:disabled,
  .colab-df-quickchart-complete:disabled:hover {
    background-color: var(--disabled-bg-color);
    fill: var(--disabled-fill-color);
    box-shadow: none;
  }

  .colab-df-spinner {
    border: 2px solid var(--fill-color);
    border-color: transparent;
    border-bottom-color: var(--fill-color);
    animation:
      spin 1s steps(1) infinite;
  }

  @keyframes spin {
    0% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
      border-left-color: var(--fill-color);
    }
    20% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    30% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
      border-right-color: var(--fill-color);
    }
    40% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    60% {
      border-color: transparent;
      border-right-color: var(--fill-color);
    }
    80% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-bottom-color: var(--fill-color);
    }
    90% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
    }
  }
</style>

  <script>
    async function quickchart(key) {
      const quickchartButtonEl =
        document.querySelector('#' + key + ' button');
      quickchartButtonEl.disabled = true;  // To prevent multiple clicks.
      quickchartButtonEl.classList.add('colab-df-spinner');
      try {
        const charts = await google.colab.kernel.invokeFunction(
            'suggestCharts', [key], {});
      } catch (error) {
        console.error('Error during call to suggestCharts:', error);
      }
      quickchartButtonEl.classList.remove('colab-df-spinner');
      quickchartButtonEl.classList.add('colab-df-quickchart-complete');
    }
    (() => {
      let quickchartButtonEl =
        document.querySelector('#df-b1505a36-905d-4651-bcc1-027ae9d6f01a button');
      quickchartButtonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';
    })();
  </script>
</div>

    </div>
  </div>





```python
data.shape
```




    (208, 61)




```python
data.describe()
```





  <div id="df-b88565bb-fb16-49b6-92f0-5c1d6dad9471" class="colab-df-container">
    <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
      <th>4</th>
      <th>5</th>
      <th>6</th>
      <th>7</th>
      <th>8</th>
      <th>9</th>
      <th>...</th>
      <th>50</th>
      <th>51</th>
      <th>52</th>
      <th>53</th>
      <th>54</th>
      <th>55</th>
      <th>56</th>
      <th>57</th>
      <th>58</th>
      <th>59</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>208.000000</td>
      <td>208.000000</td>
      <td>208.000000</td>
      <td>208.000000</td>
      <td>208.000000</td>
      <td>208.000000</td>
      <td>208.000000</td>
      <td>208.000000</td>
      <td>208.000000</td>
      <td>208.000000</td>
      <td>...</td>
      <td>208.000000</td>
      <td>208.000000</td>
      <td>208.000000</td>
      <td>208.000000</td>
      <td>208.000000</td>
      <td>208.000000</td>
      <td>208.000000</td>
      <td>208.000000</td>
      <td>208.000000</td>
      <td>208.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>0.029164</td>
      <td>0.038437</td>
      <td>0.043832</td>
      <td>0.053892</td>
      <td>0.075202</td>
      <td>0.104570</td>
      <td>0.121747</td>
      <td>0.134799</td>
      <td>0.178003</td>
      <td>0.208259</td>
      <td>...</td>
      <td>0.016069</td>
      <td>0.013420</td>
      <td>0.010709</td>
      <td>0.010941</td>
      <td>0.009290</td>
      <td>0.008222</td>
      <td>0.007820</td>
      <td>0.007949</td>
      <td>0.007941</td>
      <td>0.006507</td>
    </tr>
    <tr>
      <th>std</th>
      <td>0.022991</td>
      <td>0.032960</td>
      <td>0.038428</td>
      <td>0.046528</td>
      <td>0.055552</td>
      <td>0.059105</td>
      <td>0.061788</td>
      <td>0.085152</td>
      <td>0.118387</td>
      <td>0.134416</td>
      <td>...</td>
      <td>0.012008</td>
      <td>0.009634</td>
      <td>0.007060</td>
      <td>0.007301</td>
      <td>0.007088</td>
      <td>0.005736</td>
      <td>0.005785</td>
      <td>0.006470</td>
      <td>0.006181</td>
      <td>0.005031</td>
    </tr>
    <tr>
      <th>min</th>
      <td>0.001500</td>
      <td>0.000600</td>
      <td>0.001500</td>
      <td>0.005800</td>
      <td>0.006700</td>
      <td>0.010200</td>
      <td>0.003300</td>
      <td>0.005500</td>
      <td>0.007500</td>
      <td>0.011300</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.000800</td>
      <td>0.000500</td>
      <td>0.001000</td>
      <td>0.000600</td>
      <td>0.000400</td>
      <td>0.000300</td>
      <td>0.000300</td>
      <td>0.000100</td>
      <td>0.000600</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>0.013350</td>
      <td>0.016450</td>
      <td>0.018950</td>
      <td>0.024375</td>
      <td>0.038050</td>
      <td>0.067025</td>
      <td>0.080900</td>
      <td>0.080425</td>
      <td>0.097025</td>
      <td>0.111275</td>
      <td>...</td>
      <td>0.008425</td>
      <td>0.007275</td>
      <td>0.005075</td>
      <td>0.005375</td>
      <td>0.004150</td>
      <td>0.004400</td>
      <td>0.003700</td>
      <td>0.003600</td>
      <td>0.003675</td>
      <td>0.003100</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>0.022800</td>
      <td>0.030800</td>
      <td>0.034300</td>
      <td>0.044050</td>
      <td>0.062500</td>
      <td>0.092150</td>
      <td>0.106950</td>
      <td>0.112100</td>
      <td>0.152250</td>
      <td>0.182400</td>
      <td>...</td>
      <td>0.013900</td>
      <td>0.011400</td>
      <td>0.009550</td>
      <td>0.009300</td>
      <td>0.007500</td>
      <td>0.006850</td>
      <td>0.005950</td>
      <td>0.005800</td>
      <td>0.006400</td>
      <td>0.005300</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>0.035550</td>
      <td>0.047950</td>
      <td>0.057950</td>
      <td>0.064500</td>
      <td>0.100275</td>
      <td>0.134125</td>
      <td>0.154000</td>
      <td>0.169600</td>
      <td>0.233425</td>
      <td>0.268700</td>
      <td>...</td>
      <td>0.020825</td>
      <td>0.016725</td>
      <td>0.014900</td>
      <td>0.014500</td>
      <td>0.012100</td>
      <td>0.010575</td>
      <td>0.010425</td>
      <td>0.010350</td>
      <td>0.010325</td>
      <td>0.008525</td>
    </tr>
    <tr>
      <th>max</th>
      <td>0.137100</td>
      <td>0.233900</td>
      <td>0.305900</td>
      <td>0.426400</td>
      <td>0.401000</td>
      <td>0.382300</td>
      <td>0.372900</td>
      <td>0.459000</td>
      <td>0.682800</td>
      <td>0.710600</td>
      <td>...</td>
      <td>0.100400</td>
      <td>0.070900</td>
      <td>0.039000</td>
      <td>0.035200</td>
      <td>0.044700</td>
      <td>0.039400</td>
      <td>0.035500</td>
      <td>0.044000</td>
      <td>0.036400</td>
      <td>0.043900</td>
    </tr>
  </tbody>
</table>
<p>8 rows × 60 columns</p>
</div>
    <div class="colab-df-buttons">

  <div class="colab-df-container">
    <button class="colab-df-convert" onclick="convertToInteractive('df-b88565bb-fb16-49b6-92f0-5c1d6dad9471')"
            title="Convert this dataframe to an interactive table."
            style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px" viewBox="0 -960 960 960">
    <path d="M120-120v-720h720v720H120Zm60-500h600v-160H180v160Zm220 220h160v-160H400v160Zm0 220h160v-160H400v160ZM180-400h160v-160H180v160Zm440 0h160v-160H620v160ZM180-180h160v-160H180v160Zm440 0h160v-160H620v160Z"/>
  </svg>
    </button>

  <style>
    .colab-df-container {
      display:flex;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    .colab-df-buttons div {
      margin-bottom: 4px;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

    <script>
      const buttonEl =
        document.querySelector('#df-b88565bb-fb16-49b6-92f0-5c1d6dad9471 button.colab-df-convert');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      async function convertToInteractive(key) {
        const element = document.querySelector('#df-b88565bb-fb16-49b6-92f0-5c1d6dad9471');
        const dataTable =
          await google.colab.kernel.invokeFunction('convertToInteractive',
                                                    [key], {});
        if (!dataTable) return;

        const docLinkHtml = 'Like what you see? Visit the ' +
          '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
          + ' to learn more about interactive tables.';
        element.innerHTML = '';
        dataTable['output_type'] = 'display_data';
        await google.colab.output.renderOutput(dataTable, element);
        const docLink = document.createElement('div');
        docLink.innerHTML = docLinkHtml;
        element.appendChild(docLink);
      }
    </script>
  </div>


<div id="df-ed64f2e1-0c8c-40ec-913b-8ea9636033d5">
  <button class="colab-df-quickchart" onclick="quickchart('df-ed64f2e1-0c8c-40ec-913b-8ea9636033d5')"
            title="Suggest charts"
            style="display:none;">

<svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
     width="24px">
    <g>
        <path d="M19 3H5c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h14c1.1 0 2-.9 2-2V5c0-1.1-.9-2-2-2zM9 17H7v-7h2v7zm4 0h-2V7h2v10zm4 0h-2v-4h2v4z"/>
    </g>
</svg>
  </button>

<style>
  .colab-df-quickchart {
      --bg-color: #E8F0FE;
      --fill-color: #1967D2;
      --hover-bg-color: #E2EBFA;
      --hover-fill-color: #174EA6;
      --disabled-fill-color: #AAA;
      --disabled-bg-color: #DDD;
  }

  [theme=dark] .colab-df-quickchart {
      --bg-color: #3B4455;
      --fill-color: #D2E3FC;
      --hover-bg-color: #434B5C;
      --hover-fill-color: #FFFFFF;
      --disabled-bg-color: #3B4455;
      --disabled-fill-color: #666;
  }

  .colab-df-quickchart {
    background-color: var(--bg-color);
    border: none;
    border-radius: 50%;
    cursor: pointer;
    display: none;
    fill: var(--fill-color);
    height: 32px;
    padding: 0;
    width: 32px;
  }

  .colab-df-quickchart:hover {
    background-color: var(--hover-bg-color);
    box-shadow: 0 1px 2px rgba(60, 64, 67, 0.3), 0 1px 3px 1px rgba(60, 64, 67, 0.15);
    fill: var(--button-hover-fill-color);
  }

  .colab-df-quickchart-complete:disabled,
  .colab-df-quickchart-complete:disabled:hover {
    background-color: var(--disabled-bg-color);
    fill: var(--disabled-fill-color);
    box-shadow: none;
  }

  .colab-df-spinner {
    border: 2px solid var(--fill-color);
    border-color: transparent;
    border-bottom-color: var(--fill-color);
    animation:
      spin 1s steps(1) infinite;
  }

  @keyframes spin {
    0% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
      border-left-color: var(--fill-color);
    }
    20% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    30% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
      border-right-color: var(--fill-color);
    }
    40% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    60% {
      border-color: transparent;
      border-right-color: var(--fill-color);
    }
    80% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-bottom-color: var(--fill-color);
    }
    90% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
    }
  }
</style>

  <script>
    async function quickchart(key) {
      const quickchartButtonEl =
        document.querySelector('#' + key + ' button');
      quickchartButtonEl.disabled = true;  // To prevent multiple clicks.
      quickchartButtonEl.classList.add('colab-df-spinner');
      try {
        const charts = await google.colab.kernel.invokeFunction(
            'suggestCharts', [key], {});
      } catch (error) {
        console.error('Error during call to suggestCharts:', error);
      }
      quickchartButtonEl.classList.remove('colab-df-spinner');
      quickchartButtonEl.classList.add('colab-df-quickchart-complete');
    }
    (() => {
      let quickchartButtonEl =
        document.querySelector('#df-ed64f2e1-0c8c-40ec-913b-8ea9636033d5 button');
      quickchartButtonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';
    })();
  </script>
</div>

    </div>
  </div>





```python
data[60].value_counts()
```




    60
    M    111
    R     97
    Name: count, dtype: int64




```python
data.groupby(60).mean()
```





  <div id="df-93068de8-66f1-4946-8c02-aad7e4a1e8e2" class="colab-df-container">
    <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
      <th>4</th>
      <th>5</th>
      <th>6</th>
      <th>7</th>
      <th>8</th>
      <th>9</th>
      <th>...</th>
      <th>50</th>
      <th>51</th>
      <th>52</th>
      <th>53</th>
      <th>54</th>
      <th>55</th>
      <th>56</th>
      <th>57</th>
      <th>58</th>
      <th>59</th>
    </tr>
    <tr>
      <th>60</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>M</th>
      <td>0.034989</td>
      <td>0.045544</td>
      <td>0.050720</td>
      <td>0.064768</td>
      <td>0.086715</td>
      <td>0.111864</td>
      <td>0.128359</td>
      <td>0.149832</td>
      <td>0.213492</td>
      <td>0.251022</td>
      <td>...</td>
      <td>0.019352</td>
      <td>0.016014</td>
      <td>0.011643</td>
      <td>0.012185</td>
      <td>0.009923</td>
      <td>0.008914</td>
      <td>0.007825</td>
      <td>0.009060</td>
      <td>0.008695</td>
      <td>0.006930</td>
    </tr>
    <tr>
      <th>R</th>
      <td>0.022498</td>
      <td>0.030303</td>
      <td>0.035951</td>
      <td>0.041447</td>
      <td>0.062028</td>
      <td>0.096224</td>
      <td>0.114180</td>
      <td>0.117596</td>
      <td>0.137392</td>
      <td>0.159325</td>
      <td>...</td>
      <td>0.012311</td>
      <td>0.010453</td>
      <td>0.009640</td>
      <td>0.009518</td>
      <td>0.008567</td>
      <td>0.007430</td>
      <td>0.007814</td>
      <td>0.006677</td>
      <td>0.007078</td>
      <td>0.006024</td>
    </tr>
  </tbody>
</table>
<p>2 rows × 60 columns</p>
</div>
    <div class="colab-df-buttons">

  <div class="colab-df-container">
    <button class="colab-df-convert" onclick="convertToInteractive('df-93068de8-66f1-4946-8c02-aad7e4a1e8e2')"
            title="Convert this dataframe to an interactive table."
            style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px" viewBox="0 -960 960 960">
    <path d="M120-120v-720h720v720H120Zm60-500h600v-160H180v160Zm220 220h160v-160H400v160Zm0 220h160v-160H400v160ZM180-400h160v-160H180v160Zm440 0h160v-160H620v160ZM180-180h160v-160H180v160Zm440 0h160v-160H620v160Z"/>
  </svg>
    </button>

  <style>
    .colab-df-container {
      display:flex;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    .colab-df-buttons div {
      margin-bottom: 4px;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

    <script>
      const buttonEl =
        document.querySelector('#df-93068de8-66f1-4946-8c02-aad7e4a1e8e2 button.colab-df-convert');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      async function convertToInteractive(key) {
        const element = document.querySelector('#df-93068de8-66f1-4946-8c02-aad7e4a1e8e2');
        const dataTable =
          await google.colab.kernel.invokeFunction('convertToInteractive',
                                                    [key], {});
        if (!dataTable) return;

        const docLinkHtml = 'Like what you see? Visit the ' +
          '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
          + ' to learn more about interactive tables.';
        element.innerHTML = '';
        dataTable['output_type'] = 'display_data';
        await google.colab.output.renderOutput(dataTable, element);
        const docLink = document.createElement('div');
        docLink.innerHTML = docLinkHtml;
        element.appendChild(docLink);
      }
    </script>
  </div>


<div id="df-02ded5f3-b78a-43b0-9bd6-250911ae5c9f">
  <button class="colab-df-quickchart" onclick="quickchart('df-02ded5f3-b78a-43b0-9bd6-250911ae5c9f')"
            title="Suggest charts"
            style="display:none;">

<svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
     width="24px">
    <g>
        <path d="M19 3H5c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h14c1.1 0 2-.9 2-2V5c0-1.1-.9-2-2-2zM9 17H7v-7h2v7zm4 0h-2V7h2v10zm4 0h-2v-4h2v4z"/>
    </g>
</svg>
  </button>

<style>
  .colab-df-quickchart {
      --bg-color: #E8F0FE;
      --fill-color: #1967D2;
      --hover-bg-color: #E2EBFA;
      --hover-fill-color: #174EA6;
      --disabled-fill-color: #AAA;
      --disabled-bg-color: #DDD;
  }

  [theme=dark] .colab-df-quickchart {
      --bg-color: #3B4455;
      --fill-color: #D2E3FC;
      --hover-bg-color: #434B5C;
      --hover-fill-color: #FFFFFF;
      --disabled-bg-color: #3B4455;
      --disabled-fill-color: #666;
  }

  .colab-df-quickchart {
    background-color: var(--bg-color);
    border: none;
    border-radius: 50%;
    cursor: pointer;
    display: none;
    fill: var(--fill-color);
    height: 32px;
    padding: 0;
    width: 32px;
  }

  .colab-df-quickchart:hover {
    background-color: var(--hover-bg-color);
    box-shadow: 0 1px 2px rgba(60, 64, 67, 0.3), 0 1px 3px 1px rgba(60, 64, 67, 0.15);
    fill: var(--button-hover-fill-color);
  }

  .colab-df-quickchart-complete:disabled,
  .colab-df-quickchart-complete:disabled:hover {
    background-color: var(--disabled-bg-color);
    fill: var(--disabled-fill-color);
    box-shadow: none;
  }

  .colab-df-spinner {
    border: 2px solid var(--fill-color);
    border-color: transparent;
    border-bottom-color: var(--fill-color);
    animation:
      spin 1s steps(1) infinite;
  }

  @keyframes spin {
    0% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
      border-left-color: var(--fill-color);
    }
    20% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    30% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
      border-right-color: var(--fill-color);
    }
    40% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    60% {
      border-color: transparent;
      border-right-color: var(--fill-color);
    }
    80% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-bottom-color: var(--fill-color);
    }
    90% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
    }
  }
</style>

  <script>
    async function quickchart(key) {
      const quickchartButtonEl =
        document.querySelector('#' + key + ' button');
      quickchartButtonEl.disabled = true;  // To prevent multiple clicks.
      quickchartButtonEl.classList.add('colab-df-spinner');
      try {
        const charts = await google.colab.kernel.invokeFunction(
            'suggestCharts', [key], {});
      } catch (error) {
        console.error('Error during call to suggestCharts:', error);
      }
      quickchartButtonEl.classList.remove('colab-df-spinner');
      quickchartButtonEl.classList.add('colab-df-quickchart-complete');
    }
    (() => {
      let quickchartButtonEl =
        document.querySelector('#df-02ded5f3-b78a-43b0-9bd6-250911ae5c9f button');
      quickchartButtonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';
    })();
  </script>
</div>

    </div>
  </div>





```python
X = data.drop(columns=60, axis=1)
Y =data[60]
print(X)
print(Y)
```

             0       1       2       3       4       5       6       7       8   \
    0    0.0200  0.0371  0.0428  0.0207  0.0954  0.0986  0.1539  0.1601  0.3109   
    1    0.0453  0.0523  0.0843  0.0689  0.1183  0.2583  0.2156  0.3481  0.3337   
    2    0.0262  0.0582  0.1099  0.1083  0.0974  0.2280  0.2431  0.3771  0.5598   
    3    0.0100  0.0171  0.0623  0.0205  0.0205  0.0368  0.1098  0.1276  0.0598   
    4    0.0762  0.0666  0.0481  0.0394  0.0590  0.0649  0.1209  0.2467  0.3564   
    ..      ...     ...     ...     ...     ...     ...     ...     ...     ...   
    203  0.0187  0.0346  0.0168  0.0177  0.0393  0.1630  0.2028  0.1694  0.2328   
    204  0.0323  0.0101  0.0298  0.0564  0.0760  0.0958  0.0990  0.1018  0.1030   
    205  0.0522  0.0437  0.0180  0.0292  0.0351  0.1171  0.1257  0.1178  0.1258   
    206  0.0303  0.0353  0.0490  0.0608  0.0167  0.1354  0.1465  0.1123  0.1945   
    207  0.0260  0.0363  0.0136  0.0272  0.0214  0.0338  0.0655  0.1400  0.1843   
    
             9   ...      50      51      52      53      54      55      56  \
    0    0.2111  ...  0.0232  0.0027  0.0065  0.0159  0.0072  0.0167  0.0180   
    1    0.2872  ...  0.0125  0.0084  0.0089  0.0048  0.0094  0.0191  0.0140   
    2    0.6194  ...  0.0033  0.0232  0.0166  0.0095  0.0180  0.0244  0.0316   
    3    0.1264  ...  0.0241  0.0121  0.0036  0.0150  0.0085  0.0073  0.0050   
    4    0.4459  ...  0.0156  0.0031  0.0054  0.0105  0.0110  0.0015  0.0072   
    ..      ...  ...     ...     ...     ...     ...     ...     ...     ...   
    203  0.2684  ...  0.0203  0.0116  0.0098  0.0199  0.0033  0.0101  0.0065   
    204  0.2154  ...  0.0051  0.0061  0.0093  0.0135  0.0063  0.0063  0.0034   
    205  0.2529  ...  0.0155  0.0160  0.0029  0.0051  0.0062  0.0089  0.0140   
    206  0.2354  ...  0.0042  0.0086  0.0046  0.0126  0.0036  0.0035  0.0034   
    207  0.2354  ...  0.0181  0.0146  0.0129  0.0047  0.0039  0.0061  0.0040   
    
             57      58      59  
    0    0.0084  0.0090  0.0032  
    1    0.0049  0.0052  0.0044  
    2    0.0164  0.0095  0.0078  
    3    0.0044  0.0040  0.0117  
    4    0.0048  0.0107  0.0094  
    ..      ...     ...     ...  
    203  0.0115  0.0193  0.0157  
    204  0.0032  0.0062  0.0067  
    205  0.0138  0.0077  0.0031  
    206  0.0079  0.0036  0.0048  
    207  0.0036  0.0061  0.0115  
    
    [208 rows x 60 columns]
    0      R
    1      R
    2      R
    3      R
    4      R
          ..
    203    M
    204    M
    205    M
    206    M
    207    M
    Name: 60, Length: 208, dtype: object


Training and Test Data Split


```python
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.1, stratify=Y, random_state=1)
print(X.shape,X_train.shape,X_test.shape)
```

    (208, 60) (187, 60) (21, 60)


Model Training


```python
model = LogisticRegression()
```


```python
model.fit(X_train, Y_train)
```




<style>#sk-container-id-1 {color: black;background-color: white;}#sk-container-id-1 pre{padding: 0;}#sk-container-id-1 div.sk-toggleable {background-color: white;}#sk-container-id-1 label.sk-toggleable__label {cursor: pointer;display: block;width: 100%;margin-bottom: 0;padding: 0.3em;box-sizing: border-box;text-align: center;}#sk-container-id-1 label.sk-toggleable__label-arrow:before {content: "▸";float: left;margin-right: 0.25em;color: #696969;}#sk-container-id-1 label.sk-toggleable__label-arrow:hover:before {color: black;}#sk-container-id-1 div.sk-estimator:hover label.sk-toggleable__label-arrow:before {color: black;}#sk-container-id-1 div.sk-toggleable__content {max-height: 0;max-width: 0;overflow: hidden;text-align: left;background-color: #f0f8ff;}#sk-container-id-1 div.sk-toggleable__content pre {margin: 0.2em;color: black;border-radius: 0.25em;background-color: #f0f8ff;}#sk-container-id-1 input.sk-toggleable__control:checked~div.sk-toggleable__content {max-height: 200px;max-width: 100%;overflow: auto;}#sk-container-id-1 input.sk-toggleable__control:checked~label.sk-toggleable__label-arrow:before {content: "▾";}#sk-container-id-1 div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-1 div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-1 input.sk-hidden--visually {border: 0;clip: rect(1px 1px 1px 1px);clip: rect(1px, 1px, 1px, 1px);height: 1px;margin: -1px;overflow: hidden;padding: 0;position: absolute;width: 1px;}#sk-container-id-1 div.sk-estimator {font-family: monospace;background-color: #f0f8ff;border: 1px dotted black;border-radius: 0.25em;box-sizing: border-box;margin-bottom: 0.5em;}#sk-container-id-1 div.sk-estimator:hover {background-color: #d4ebff;}#sk-container-id-1 div.sk-parallel-item::after {content: "";width: 100%;border-bottom: 1px solid gray;flex-grow: 1;}#sk-container-id-1 div.sk-label:hover label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-1 div.sk-serial::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: 0;}#sk-container-id-1 div.sk-serial {display: flex;flex-direction: column;align-items: center;background-color: white;padding-right: 0.2em;padding-left: 0.2em;position: relative;}#sk-container-id-1 div.sk-item {position: relative;z-index: 1;}#sk-container-id-1 div.sk-parallel {display: flex;align-items: stretch;justify-content: center;background-color: white;position: relative;}#sk-container-id-1 div.sk-item::before, #sk-container-id-1 div.sk-parallel-item::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: -1;}#sk-container-id-1 div.sk-parallel-item {display: flex;flex-direction: column;z-index: 1;position: relative;background-color: white;}#sk-container-id-1 div.sk-parallel-item:first-child::after {align-self: flex-end;width: 50%;}#sk-container-id-1 div.sk-parallel-item:last-child::after {align-self: flex-start;width: 50%;}#sk-container-id-1 div.sk-parallel-item:only-child::after {width: 0;}#sk-container-id-1 div.sk-dashed-wrapped {border: 1px dashed gray;margin: 0 0.4em 0.5em 0.4em;box-sizing: border-box;padding-bottom: 0.4em;background-color: white;}#sk-container-id-1 div.sk-label label {font-family: monospace;font-weight: bold;display: inline-block;line-height: 1.2em;}#sk-container-id-1 div.sk-label-container {text-align: center;}#sk-container-id-1 div.sk-container {/* jupyter's `normalize.less` sets `[hidden] { display: none; }` but bootstrap.min.css set `[hidden] { display: none !important; }` so we also need the `!important` here to be able to override the default hidden behavior on the sphinx rendered scikit-learn.org. See: https://github.com/scikit-learn/scikit-learn/issues/21755 */display: inline-block !important;position: relative;}#sk-container-id-1 div.sk-text-repr-fallback {display: none;}</style><div id="sk-container-id-1" class="sk-top-container"><div class="sk-text-repr-fallback"><pre>LogisticRegression()</pre><b>In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook. <br />On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.</b></div><div class="sk-container" hidden><div class="sk-item"><div class="sk-estimator sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-1" type="checkbox" checked><label for="sk-estimator-id-1" class="sk-toggleable__label sk-toggleable__label-arrow">LogisticRegression</label><div class="sk-toggleable__content"><pre>LogisticRegression()</pre></div></div></div></div></div>



Model Evaluation


```python
X_train_prediction = model.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction, Y_train)
```


```python
print(training_data_accuracy)
```

    0.8342245989304813



```python
X_test_prediction = model.predict(X_test)
testing_data_accuracy = accuracy_score(X_test_prediction, Y_test)
```


```python
print(testing_data_accuracy)
```

    0.7619047619047619


Predicition System


```python
input_data = (0.0100,0.0171,0.0623,0.0205,0.0205,0.0368,0.1098,0.1276,0.0598,0.1264,0.0881,0.1992,0.0184,0.2261,0.1729,0.2131,0.0693,0.2281,0.4060,0.3973,0.2741,0.3690,0.5556,0.4846,0.3140,0.5334,0.5256,0.2520,0.2090,0.3559,0.6260,0.7340,0.6120,0.3497,0.3953,0.3012,0.5408,0.8814,0.9857,0.9167,0.6121,0.5006,0.3210,0.3202,0.4295,0.3654,0.2655,0.1576,0.0681,0.0294,0.0241,0.0121,0.0036,0.0150,0.0085,0.0073,0.0050,0.0044,0.0040,0.0117)
input_data_as_numpy_array = np.asarray(input_data)
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)
prediction = model.predict(input_data_reshaped)

if (prediction[0] == 'M'):
  print('The Object is a Mine')

```

    The Object is a Mine



```python

```


```python

```
