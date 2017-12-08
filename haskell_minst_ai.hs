import Control.Monad
import Data.Functor
import System.Random
import Codec.Compression.GZip (decompress)
import qualified Data.ByteString.Lazy as BS
import Data.Ord
import Data.List

--Box-Muller transformation to generate a random number between 0 and scale
boxMuller scale = do
	x1 <- randomIO
	x2 <- randomIO
	return $ scale * sqrt (-2 * log x1) * cos (2 * pi * x2)


{-|
[[([Float], [[Float]])]] represents the neural network
The bias for the input to the ith neuron is represented by the ith float in [Float]
The weight for the input to the ith node from the jth input is represented by the jth float in the ith row of [[Float]] 
-}
neuralNetwork :: [Int] -> IO [([Float], [[Float]])]
neuralNetwork size@(_:ts) = zip (flip replicate 1 <$> ts) <$>
	zipWithM (\m n -> replicateM n $ replicateM m $ boxMuller 0.01) size ts

--Rectifier activation function
relu = max 0
--Derivative of activation function
drelu x | x < 0	= 0
	| otherwise	= 1

--Computes the next layer from the previous layer by passing the values through weights and biases
nextLayer :: [Float] -> ([Float], [[Float]]) -> [Float]
nextLayer vActivation (vBias, vsWeight) = zipWith (+) vBias $ sum . zipWith (*) vActivation <$> vsWeight

--Runs the activation function on the sum of the weighted inputs
feed :: [Float] -> [([Float], [[Float]])] -> [Float]
feed = foldl' (((relu <$>) . ) . nextLayer)


{-|
vX: vector containing the inputs
Returns a list of (weighted inputs, activations) for each layer in reverse order
-}
revaz vX = foldl' (\(avs@(av:_), zs) (bs, wms) -> let
	zs' = nextLayer av (bs, wms) in ((relu <$> zs'):avs, zs':zs)) ([vX], [])

--Derivative of cost function
dCost a y | y == 1 && a >= y = 0
	| otherwise	= a - y


{-|
xv: vector of inputs
yv: vector of correct outputs
Return list (activations, delta) for each layer
-}
deltas :: [Float] -> [Float] -> [([Float], [[Float]])] -> ([[Float]], [[Float]])
deltas xv yv layers = let
	(avs@(av:_), zv:zvs) = revaz xv layers
	delta0 = zipWith (*) (zipWith dCost av yv) (drelu <$> zv)
	in (reverse avs, f (transpose . snd <$> reverse layers) zvs [delta0]) where
		f _ [] dvs = dvs
		f (wm:wms) (zv:zvs) dvs@(dv:_) = f wms zvs $ (:dvs) $
			zipWith (*) [sum $ zipWith (*) row dv | row <- wm] (drelu <$> zv)

--rate of learning
eta = 0.002

descend :: [Float] -> [Float] -> [Float]
descend av dv = zipWith (-) av ((eta *) <$> dv)

learn xv yv layers = let (avs, dvs) = deltas xv yv layers
	in zip (zipWith descend (fst <$> layers) dvs) $
	zipWith3 (\wvs av dv -> zipWith (\wv d -> descend wv ((d*) <$> av)) wvs dv)
	(snd <$> layers) avs dvs

getImage s n = fromIntegral . BS.index s . (n*28^2 + 16 +) <$> [0..28^2 - 1]
getX	 s n = (/ 256) <$> getImage s n
getLabel s n = fromIntegral $ BS.index s (n + 8)
getY	 s n = fromIntegral . fromEnum . (getLabel s n ==) <$> [0..9]

render n = let s = " .:oO@" in s !! (fromIntegral n * length s `div` 256)

main = do
	[trainI, trainL, testI, testL] <- mapM ((decompress	<$>) . BS.readFile)
		[ "train-images.gz"
		, "train-labels.gz"
		,	"test-images.gz"
		,	"test-labels.gz"
		]
	b <- neuralNetwork [784, 70, 10]
	n <- (`mod` 10000) <$> randomIO
	putStr . unlines $
		take 28 $ take 28 <$> iterate (drop 28) (render <$> getImage testI n)

	let
		example = getX testI n
		bs = scanl (foldl' (\b n -> learn (getX trainI n) (getY trainL n) b)) b [
	 		[	 0.. 999],
	 		[1000..2999],
	 		[3000..5999],
	 		[6000..9999]]
		smart = last bs
		cute d score = show d ++ ": " ++ replicate (round $ 70 * min 1 score) '+'
		bestOf = fst . maximumBy (comparing snd) . zip [0..]

	forM_ bs $ putStrLn . unlines . zipWith cute [0..9] . feed example

	putStrLn $ "best guess: " ++ show (bestOf $ feed example smart)

	let guesses = bestOf . (\n -> feed (getX testI n) smart) <$> [0..9999]
	let answers = getLabel testL <$> [0..9999]
	putStrLn $ show (sum $ fromEnum <$> zipWith (==) guesses answers) ++ " / 10000"