import numpy as np
from skimage import io, graph, segmentation, morphology, util
import tqdm
import random
from .packing import *

# Class for storing diamond shape patches
class Patch:
    patch: np.ndarray
    overlap: int
    size: int
    
    def __init__(self, src, center_y, center_x, size, overlap):
        self.overlap = overlap
        self.size = size
        l = int(size/2) + overlap
        self.patch = util.img_as_float( src[(center_y-l):(center_y+l), (center_x-l):(center_x+l), :] )

    # Crop the patch for the North edge, masked by an optional mask.
    def N(self, mask=None):
        result = np.zeros([self.size, self.size, 3])
        result[0:int(self.size/2+self.overlap),:,:] = self.patch[int(self.size/2+self.overlap):(self.size+2*self.overlap),self.overlap:(self.size+self.overlap),:]
        if(mask is None):
            mask = self.N_mask()
        result[~mask] = 0
        return result

    # Crop the patch for the South edge, masked by an optional mask.
    def S(self, mask=None):
        result = np.zeros([self.size, self.size, 3])
        result[int(self.size/2-self.overlap):self.size,:,:] = self.patch[0:int(self.size/2+self.overlap),self.overlap:(self.size+self.overlap),:]
        if(mask is None):
            mask = self.S_mask()
        result[~mask] = 0
        return result

    # Crop the patch for the West edge, masked by an optional mask.
    def W(self, mask=None):
        result = np.zeros([self.size, self.size, 3])
        result[:,0:int(self.size/2+self.overlap),:] = self.patch[self.overlap:(self.size+self.overlap),int(self.size/2+self.overlap):(self.size+2*self.overlap),:]
        if(mask is None):
            mask = self.W_mask()
        result[~mask] = 0
        return result

    # Crop the patch for the East edge, masked by an optional mask.
    def E(self, mask=None):
        result = np.zeros([self.size, self.size, 3])
        result[:,int(self.size/2-self.overlap):self.size,:] = self.patch[self.overlap:(self.size+self.overlap),0:int(self.size/2+self.overlap),:]
        if(mask is None):
            mask = self.E_mask()        
        result[~mask] = 0
        return result

    # Default North edge mask
    def N_mask(self):
        a = np.arange(self.size)
        C = np.empty([self.size,self.size,2])
        C[:,:,0] = a
        C[:,:,1] = a[:,None]
        return (C[:,:,0] > C[:,:,1]-self.overlap) & (self.size - 1 - C[:,:,0] > C[:,:,1]-self.overlap)
        
    # Default South edge mask
    def S_mask(self):
        a = np.arange(self.size)
        C = np.empty([self.size,self.size,2])
        C[:,:,0] = a
        C[:,:,1] = a[:,None]
        return (C[:,:,0] < C[:,:,1]+self.overlap) & (self.size - 1 - C[:,:,0] < C[:,:,1]+self.overlap)

    # Default West edge mask
    def W_mask(self):
        a = np.arange(self.size)
        C = np.empty([self.size,self.size,2])
        C[:,:,0] = a
        C[:,:,1] = a[:,None]
        return (C[:,:,1] > C[:,:,0]-self.overlap) & (self.size - 1 - C[:,:,1] > C[:,:,0]-self.overlap)

    # Default East edge mask
    def E_mask(self):
        a = np.arange(self.size)
        C = np.empty([self.size,self.size,2])
        C[:,:,0] = a
        C[:,:,1] = a[:,None]
        return (C[:,:,1] < C[:,:,0]+self.overlap) & (self.size - 1 - C[:,:,1] < C[:,:,0]+self.overlap)
    

# Merge N,W,S, and E pacthes 
def merge_patches(N, W, S, E, visulize_edges=False, visulize_error=False):

    # Create error image
    NW_mask = N.N_mask() & W.W_mask()
    NS_mask = N.N_mask() & S.S_mask()
    NE_mask = N.N_mask() & E.E_mask()
    WS_mask = W.W_mask() & S.S_mask()
    WE_mask = W.W_mask() & E.E_mask()
    SE_mask = S.S_mask() & E.E_mask()
    errorMap = ((N.N(NW_mask) - W.W(NW_mask))**2) + ((N.N(NS_mask) - S.S(NS_mask))**2) + ((N.N(NE_mask) - E.E(NE_mask))**2) + ((W.W(WS_mask) - S.S(WS_mask))**2) + ((W.W(WE_mask) - E.E(WE_mask))**2) + ((S.S(SE_mask) - E.E(SE_mask))**2)
    errorResult = errorMap/np.max(errorMap)
    errorMap = np.sum(errorMap, axis=2)
    
    # Create masks for the diagonal and anti-diagonal
    diagMask = NW_mask | SE_mask
    antiMask = NE_mask | WS_mask

    # Find shortest path through diagonal and anti-diagonal
    diag = graph.MCP(costs = errorMap * diagMask - (1-diagMask), fully_connected=True)
    diag.find_costs( [[0,0]], [[-1,-1]] )
    diag_path = diag.traceback([-1,-1])

    anti = graph.MCP(costs = errorMap * antiMask - (1-antiMask), fully_connected=True)
    anti.find_costs( [[0,-1]], [[-1,0]] )
    anti_path = anti.traceback([-1,0])

    edgeMap = np.zeros(errorMap.shape)
    for i in diag_path:
        edgeMap[i] = 1

    for i in anti_path:
        edgeMap[i] = 1
        
    # Extract final N, W, S, E masks 
    size = edgeMap.shape[0]

    normalization = np.zeros([size,size,1])
    N_mask = segmentation.flood(edgeMap, (0, int(size/2)), connectivity=1)
    S_mask = segmentation.flood(edgeMap, (size-1, int(size/2)), connectivity=1)
    W_mask = segmentation.flood(edgeMap, (int(size/2), 0), connectivity=1)
    E_mask = segmentation.flood(edgeMap, (int(size/2), size-1), connectivity=1)
    
    
    while( np.any((normalization == 0)) ):
        mask = np.squeeze((normalization == 0), axis=2)
        N_mask = (morphology.binary_dilation(N_mask) & mask) | N_mask
        S_mask = (morphology.binary_dilation(S_mask) & mask) | S_mask
        W_mask = (morphology.binary_dilation(W_mask) & mask) | W_mask
        E_mask = (morphology.binary_dilation(E_mask) & mask) | E_mask
        
        normalization[N_mask & mask] += 1.0
        normalization[S_mask & mask] += 1.0
        normalization[W_mask & mask] += 1.0
        normalization[E_mask & mask] += 1.0
    
    # merge (and normalize)
    result = N.N(N_mask) + W.W(W_mask) + S.S(S_mask) + E.E(E_mask)
    result /= normalization
    if visulize_error:
        result = errorResult
    if visulize_edges:
        result = result * (1-edgeMap[:, :, None]) + edgeMap[:, :, None] * np.array([[[1.0, 0.0, 1.0]]])

    # error
    error = np.sum(errorMap * edgeMap)
    
    # Done
    return result, error


# create a texture wang tile from a give template; randomly select patches.
# Input:
#   colors: number of wang tile edge colors
#   img: template image
#   size: wang tile texture size
#   overlap: width of graph cut region.
#
# Output:
#   tiling: packed wang tile textures
#   error: total graph cut error
def create_tiling(colors, img, size, overlap, progress: tqdm.tqdm | None = None, **kwargs):

    # get packing 
    packing = create_packing(colors)

    # create patches
    horizontal = []
    vertical = []

    # fully random selection of patches
    tmp = int(size/2) + overlap
    for i in range(0, colors):
        horizontal.append( Patch(img, random.randint(tmp, img.shape[0]-tmp), random.randint(tmp, img.shape[1]-tmp), size, overlap) )
        vertical.append( Patch(img, random.randint(tmp, img.shape[0]-tmp), random.randint(tmp, img.shape[1]-tmp), size, overlap) )
    

    # create tiles
    tiling = np.zeros( [packing.shape[0] * size, packing.shape[1] * size, 3] )

    error = 0.0
    for y in range(0, packing.shape[0]):
        for x in range(0, packing.shape[1]):
            # get patching corresponding to tile edge colors
            N = vertical[ packing.N()[y,x] ]
            S = vertical[ packing.S()[y,x] ]
            W = horizontal[ packing.W()[y,x] ]
            E = horizontal[ packing.E()[y,x] ]

            # create tile
            tile, err = merge_patches( N, W, S, E, **kwargs )
            error += err
            
            # copy in tiling
            tiling[ (y*size):((y+1)*size), (x*size):((x+1)*size),: ] = tile
            if progress is not None:
                progress.update()
                

    return util.img_as_ubyte(tiling), error


# Make multiple attempts in creating a Wang Tiling
# Input:
#   colors: number of colors in the Wang Tiling
#   img: template image
#   size: wang tile texture size
#   overlap: width of the region to run a graph cut through
#   attempts: number of trials
#
# Output:
#   best_tiling: tiling with the lowest cut error
#   best_error: tiling cut error
#
def try_tilings(colors, img, size, overlap, attempts, **kwargs):

    best_error = 10e10
    best_tiling = None

    progress = tqdm.tqdm(total=attempts*colors**4)

    for i in range(0,attempts):
        res, error = create_tiling(colors, img, size, overlap, progress=progress, **kwargs)

        if(error < best_error):
            best_tiling = np.copy(res)
            best_error = error

    return best_tiling, best_error
        


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser('classicwang')
    parser.add_argument('input', help='the input image')
    parser.add_argument('output', help='the output packed tiling')
    parser.add_argument('--colors', type=int, default=3)
    parser.add_argument('--size', type=int, default=256)
    parser.add_argument('--overlap', type=int, default=10)
    parser.add_argument('--attempts', type=int, default=8)
    parser.add_argument('--visulize_edges', action='store_true')
    parser.add_argument('--visulize_error', action='store_true')
    args = parser.parse_args()
    
    img = io.imread(args.input)
    # 8-trials in creating a 3 color tiling with 256x256 textures and use a 2*10 wide overlapping band for graph cut.
    res, error = try_tilings(args.colors, img, args.size, args.overlap, args.attempts, visulize_edges=args.visulize_edges, visulize_error=args.visulize_error)
    print('ERROR:', error)
    io.imsave(args.output, res, quality=100)
