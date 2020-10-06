# Very important information

This is very very important infrmation

- qwer
- abc
- zxcv

```python
chiSq = 0
x = np.zeros (nR, dtype=np.float64)
for j in range (nsave):
    x = (1-self.leak_rate) * x + self.leak_rate * np.tanh (np.matmul(self.W_in, rI[j]) + np.matmul(self.W, x))
    tmp = np.append(x, rI[j])
    y = np.matmul(self.W_out, tmp)
    df = y - rO[j]
    chiSq += np.dot (df, df)
chiSq = chiSq / 2 / nsave
return chiSq
```
