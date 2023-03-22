---
title: Haskellで自作する言語処理系
subtitle: LLVMにGCを添えて
---

## 予定

1. Haskell入門
2. Parsecでコンパイラを作る
3. HaskellでGCを作る
4. [llvm-hs-pure](https://hackage.haskell.org/package/llvm-hs-pure)でLLVM-IRを生成
5. バイナリを生成

## Haskell

### Hello, world!

プログラムは`main`関数がエントリポイントとなる。
`print`文は、デバッグ用の標準出力。

```hs
-- single line comment

{-
 block comment
-}

main = print "hello world"
```

### ブロック

複数の文を順番に実行するには、`>>`演算子を使う。

```hs
main = print "Hello" >> print "world" >> print "yahoo";
```

また、`do`ブロックも同じ効果がある。

```hs
main = do {
  print "Hello";
  print "world";
  print "yahoo";
}
```

なお、Pythonのようにインデントを揃えると括弧が不要になる。

```hs
main = do
  print "Hello"
  print "world"
  print "yahoo"
```

### 変数

変数はimmutableである。型を明示することもできるし、型推論もできる。

```hs
foo = 114514 :: Int

bar :: Int
bar = 364364

baz = 1919810

main = do
  print foo
  print bar
  print baz
```

#### 基本型

ごく普通の何の変哲もない基本型。

```hs
bool :: Bool
bool = True

char :: Char
char = 'Z'

string :: String -- [Char]
string = "ABC"

int :: Int
int = 123

integer :: Integer -- big int
integer = 123

float :: Float -- 32bit
float = 123.45

double :: Double -- 64bit
double = 123.45

main = do
  print bool
  print char
  print string
  print int
  print integer
  print float
  print double
```

### 基本構文

ありふれた機能。

#### let文

局所変数を定義して式の中で使える。

```hs
main = print (let x = 114 in do if x == 114 then 514 else 364364)
```

#### if文

`then`を書かされる。うっかり忘れそう。`if`文の書き方、統一してほしいよね。

```hs
main = print (if True then "HOGE" else "PIYO")
```

#### case文

`switch`みたいなもの。

```hs
main = do
  x <- getLine
  case x of
    "ABC" -> print "DEF"
    "abc" -> print "def"
```

#### where文

論文書くときにwhereで数式に但し書きするでしょ。そういうイメージ。

```hs
main = do
  print (add x y)
  where
    x = 114
    y = 514
    add x y = x + y
main = print (let x = 114 in do x * x)
```

### 関数

#### 関数定義

ラムダ抽象みたいな感じ。引数は型推論できる。

```
add x y = x + y
main = print (add 1 2)
```

もちろん型を宣言できる。カリー化した形で書く。

```hs
add :: Int -> Int -> Int
add x y = x + y

main = print (add 114 514)
```

#### パターンマッチ

引数が特定の値を取るときの値を指定できる。

```hs
fact :: Int -> Int
fact 0 = 1
fact n = n * fact (n - 1)

main = print (fact 10)
```

#### ラムダ式

ラムダ抽象。複数の引数を取る場合はタプルにする必要がある。

```hs
main = print ((\(x, y) -> x + y) (114, 514))
```

#### 引数

関数適用は左結合だから、右結合にしたい時は括弧が必要だった。
`$`を使うと、括弧を省略できる。

```hs
main = do
  print $ 114 + 514 -- print (114 + 514)
  print $! 114 + 514 -- eager evaluation
```

#### 中置記法

2変数関数はバッククォートで括ると中置記法にできる。

```hs
add x y = x + y
main = do print (114 `add` 514)
```

#### 関数の結合

何が嬉しいかわからんかもだけど関数は結合できる。
覚えきれない時は愚直に`plus1(plus1 114)`で良いと思う。

```hs
plus1 x = x + 1
plus1 :: Int -> Int
print $ (plus1 . plus1) 114
```

#### 正格評価

Haskellは非正格評価...だけではなく正格評価することもできる。
そう、`seq`関数を使えばね。
引数`x`と`y`の値は`x+y`を評価する前に確定される。

```hs
add x y = x `seq` y `seq` x + y -- eager evaluation
print $ add 114 514
```

### 演算子

#### 算術演算

ぶっちゃけ`div`とか覚えてなくても困らない。

```hs
main = do
  print (11.4 + 514)
  print (11.4 - 514)
  print (11.4 * 514)
  print (11.4 / 514)
  print (114 ^ 2) -- exp (int^int)
  print (114.0 ^^ 2) -- exp(float^^int)
  print (114.0 ** 2.0) -- exp(float**float)
```

#### 比較演算

HaskellはFortranの申し子。

```hs
main = do
  print (114 == 514)
  print (114 /= 514)
  print (114 < 514)
  print (114 > 514)
  print (114 <= 514)
  print (114 >= 514)
```

#### 論理演算

```hs
main = do
  print (True && False)
  print (True || False)
  print (not True)
```

### リスト

文字列は文字のリストなんだぜ。
あと、添字は`!!`を使う。覚えにくい。
`++`演算子でリストを結合できたりする。

```hs
list :: [Char]
list = "ABCDE"

main = do
  print (list !! 2) -- 'C'
  print (list ++ "FG") -- "ABCDEFG
  print ('Z' : list) -- "ZABCDE"
  print ('B' `elem` list) -- contains
  print ('Z' `notElem` list) -- not contains
  print ([1..3])
  print ([1,2,3])
  print ([1,2,3] !! 1) -- 1番目の要素
  print ([1,2,3] ++ [4,5,6]) -- 結合
  print (length [1,2,3]) -- 長さ
  print (head [1,2,3]) -- 先頭
  print (last [1,2,3]) -- 最後
  print (init [1,2,3]) -- 末尾を除く部分リスト
  print (tail [1,2,3]) -- 先頭を除く部分リスト
  print (take 2 [1,2,3]) -- 先頭N個を抽出した部分リスト
  print (drop 2 [1,2,3]) -- 先頭N個を除去した部分リスト
  print ['A', 'B', 'C']
```

### タプル

早い話が構造体。

```hs
data Point = Point Int Int deriving Show

add (Point x1 y1) (Point x2 y2) = Point (x1 + x2) (y1 + y2)

a = Point 114 514
b = Point 364 364

main = print (add a b)
```

フィールドに名前を付けられる。

```hs
data Point = Point {x, y :: Int} deriving Show

add (Point x1 y1) (Point x2 y2) = Point (x1 + x2) (y1 + y2)

a = Point {x = 114, y = 514}
b = Point {x = 364, y = 364}

main = print (add a b)
```

### 列挙型

整数に名前を付けたもの。`deriving`で型クラスを派生させると、色々便利機能が増える。

```hs
data Actor = YJSNPI | MUR | KMR deriving (Show, Eq, Ord, Read, Enum)

actor :: Actor
actor = MUR

main = do
  print actor
  print (MUR == KMR) -- Eq class
  print (MUR <= KMR) -- Ord class
  x <- getLine
  print (read x :: Actor)
  print (fromEnum MUR :: Int) -- Enum class
  print (toEnum 1 :: Actor) -- Enum class
```

### union

共用体ではない。いわゆる直和型。

```hs
data Sexp = Atom {v :: Int} | Cons {car, cdr :: Sexp} deriving Show

main = do
  let a = Atom 114
      b = Atom 514
      c = Cons a (Cons b b) in do print c
```

### alias

型に別名を付けられる。

```hs
type Job = String
type Age = Int
type Profile = (Job, Age)

yj = ("Student", 24)
yj :: Profile
```

### 型クラス

型に関数を生やすことができる。
その関数をメソッドと呼ぶ。
個別の型引数に対してインスタンス化することもできる。

```hs
main = do print yj
class ToString v where
  toString :: v -> String

instance ToString Bool where
  toString True = "TRUE"
  toString False = "FALSE"

instance ToString String where
  toString text = text

main = do
  print $ toString True
  print $ toString False
  print $ toString "ABC"
```

### Functor

`Functor`は、`fmap`という2変数のメソッドを持つクラスである。

```hs
class Functor f where
  fmap :: (a -> b) -> f a -> f b
```

以下に実例を示す。

```hs
plus n = n + 1

main = do
  print $ fmap plus [1, 2, 3] -- [2, 3, 4]
  print $ fmap plus Nothing   -- Nothing
  print $ fmap plus (Just 5)  -- Just 6
```

要するに、他の言語の`map`関数である。

### Applicative

`Applicative`は、`Functor`の派生クラスで、`pure`と`<*>`の2個のメソッドが追加されている。

```hs
class Functor f => Applicative f where
  pure :: a -> f a
  (<*>) :: f (a -> b) -> f a -> f b
```

以下に実例を示す。

```hs
main = do
  print $ (pure (+ 1)) <*> (Just 5)  -- Just 6
  print $ (pure (+ 1)) <*> [1, 2, 3] -- [2, 3, 4]
```

要するに`fmap`とほぼ同じ機能である。
ただし、`fmap`の第1引数は裸の値だが、`<*>`演算子の左辺はファンクタにラップされた値という差がある。
`pure`は、そのファンクタの型引数を受け取って、自動的にラップしてくれる関数ということである。

### Monad

`Monad`は、`Applicative`の派生クラスで、`>>=`と`>>`と`return`の3個のメソッドが追加されている。

```hs
class Applicative m => Monad m where
  (>>=) :: m a -> (a -> m b) -> m b
  (>>) :: m a -> m b -> m b
  return :: a -> m a
```


### アクション

```hs
{-
  アクションは、副作用を持つ式のこと
-}

main = do
  x <- getLine -- get value from action
  print =<< getLine -- pass value from action to function
  getLine >>= print -- pass value from action to function
```
