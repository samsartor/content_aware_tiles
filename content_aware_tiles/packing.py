import numpy as np
import math

# Wang tiling.
# Each tile is stored as [N,W,S,E] (counter clockwise)
class WangTiling(np.ndarray):
    
    def __new__(cls, colors):
        C = int(colors*colors)
        return super().__new__(cls, shape=[C,C, 4], dtype=np.int8)

    def __init__(self, colors):
        self[:,:,:] = 0

    def colors(self):
        return int(math.sqrt(self.shape[0]))
        
    def at(self, y, x):
        return self[ y % self.shape[0], x% self.shape[1], : ]

    def N(self):
        return self[:,:,0]

    def W(self):
        return self[:,:,1]

    def S(self):
        return self[:,:,2]

    def E(self):
        return self[:,:,3]

    # returns only the W and E colors
    def horizontal(self):
        return self[:,:,1:4:2]

    # returns only the N and S colors
    def vertical(self):
        return self[:,:,0:3:2]

    # write out as CSV
    def save_csv(self, filename):
        C=self.colors()*self.colors()
        np.savetxt(filename, self.reshape(C, C*4), fmt='%d')

    # load from an CSV
    def load_csv(self, filename):
        C=self.colors()*self.colors()
        self[:,:,:] = np.loadtxt(filename).reshape(C, C, 4)

    # create from an CSV
    @classmethod
    def load_csv(filename):
        t = np.loadtxt(filename, dtype=np.int8)
        C = t.shape[0]
        tiling = WangTiling( math.sqrt(C) )
        tiling[:,:,:] = t.reshape(C, C, 4)
        return tiling

    # Write out the tiling as an SVG
    def save_svg(self, filename, colorNames=["red","green","blue","yellow","cyan"], tileSize=120, lineSize=8):
        C = self.colors()
        size = C * C * tileSize

        # create header
        f = open(filename, 'w')
        f.write("<?xml version=\"1.0\" encoding=\"UTF-8\" standalone=\"no\"?>\n")
        f.write("<!DOCTYPE svg PUBLIC \"-//W3C//DTD SVG 1.1//EN\" \"http://www.w3.org/Graphics/SVG/1.1/DTD/svg11.dtd\">\n")
        f.write("<svg width=\"" + str(size) + "\" height=\"" + str(size) + "\" viewBox=\"0 0 " + str(size) + " " + str(size) + "\">\n")

        # for each tile
        for y in range(0,C*C):
            for x in range(0,C*C):
                start_x = tileSize * x
                start_y = tileSize * y

                # write out tile
                f.write("<g>\n")
                
                # top color
                f.write("<polyline fill=\"" + colorNames[self.N()[y,x]] + "\" stroke=\"black\" points=\"")
                f.write(str(start_x) + ", " + str(start_y) + " ")
                f.write(str(start_x + lineSize) + ", " + str(start_y + lineSize) + " ")
                f.write(str(start_x + tileSize - lineSize) + ", " + str(start_y + lineSize) + " ")
                f.write(str(start_x + tileSize) + ", " + str(start_y) + "\" />\n")

                # bottom color
                f.write("<polyline fill=\"" + colorNames[self.S()[y,x]] + "\" stroke=\"black\" points=\"")
                f.write(str(start_x + lineSize) + ", " + str(start_y + tileSize - lineSize) + " ")
                f.write(str(start_x) + ", " + str(start_y + tileSize) + " ")
                f.write(str(start_x + tileSize) + ", " + str(start_y + tileSize) + " ")
                f.write(str(start_x + tileSize - lineSize) + ", " + str(start_y + tileSize - lineSize) + "\" />\n")

                # left color
                f.write("<polyline fill=\"" + colorNames[self.W()[y,x]] + "\" stroke=\"black\" points=\"")
                f.write(str(start_x) + ", " + str(start_y) + " ")
                f.write(str(start_x) + ", " + str(start_y + tileSize) + " ")
                f.write(str(start_x + lineSize) + ", " + str(start_y + tileSize - lineSize) + " ")
                f.write(str(start_x + lineSize) + ", " + str(start_y + lineSize) + "\" />\n")

                # right color
                f.write("<polyline fill=\"" + colorNames[self.E()[y,x]] + "\" stroke=\"black\" points=\"")
                f.write(str(start_x + tileSize - lineSize) + ", " + str(start_y + lineSize) + " ")
                f.write(str(start_x + tileSize - lineSize) + ", " + str(start_y + tileSize - lineSize) + " ")
                f.write(str(start_x + tileSize) + ", " + str(start_y + tileSize) + " ")
                f.write(str(start_x + tileSize) + ", " + str(start_y) + "\" />\n")

                # draw outline
                f.write("<rect fill=\"none\" stroke-width=\"1\" stroke=\"black\" x=\"" + str(start_x) + "\" y=\"" + str(start_y) + "\" width=\"" + str(tileSize) + "\" height=\"" + str(tileSize) + "\" />\n")
                
                # done tile
                f.write("</g>\n")

        # end of SVG
        f.write("</svg>\n")
        f.close()

# A domino string; a 1D sequence of 2-edge tiles.
class DominoString(np.ndarray):

    def __new__(cls, colors):
        C = int(colors*colors)
        return super().__new__(cls, shape=[C,2], dtype=np.int8)

    def __init__(self, colors):
        self[:,:] = 0

    def colors(self):
        return int(math.sqrt(self.shape[0]))

    def at(self, x):
        return self[ x % self.shape[0], : ]
    
    def N(self):
        return self[:,0]

    def W(self):
        return self.N()

    def S(self):
        return self[:,1]

    def E(self):
        return self.S()
    

# Create a complete domino string
# cf. Algorithm 1
def create_single_domino(colors):
    D = DominoString(colors)

    y = 0
    for c in range(0, colors):
        for i in range(0, c):
            D.at(y+0)[:] = [i, c]
            D.at(y+1)[:] = [c, i+1]
            y = y + 2
        D.at(y)[:] = [c, 0]
        y = y + 1

    return D

# Create two domino strings that are together twice complete and that meet thr zig-zag conditions
# cf. Algorithm 2
def create_double_domino(colors):
    D0 = DominoString(colors)
    D1 = DominoString(colors)
    
    y = 0
    for c in range(0, colors, 2):
        for i in range(0, colors):
            # First domino sequence
            # Fixed color 'c' at even edges
            D0.at(y-1)[:] = [(i+colors-1) % colors, c]
            D0.at(y+0)[:] = [c, i]
            # Second domino sequence
            # Fixed color 'c' at odd edges
            D1.at(y+0)[:] = [i, c+1]
            D1.at(y+1)[:] = [c+1, (i+1) % colors]

            y = y + 2

    return D0, D1


# Create a dual wang tile packing
def create_packing(colors):
    result = WangTiling(colors)

    r = np.arange(result.shape[0])
    rows, cols = np.ogrid[:result.shape[0], :result.shape[1]]

    # get Domino string
    D = create_single_domino(colors)
        
    # Odd colors
    if ((colors % 2) == 1):

        # assign D to horizontal (w/ positive shift) and to vertical (w/ negative shift)
        # Note: the vertical negative shift == horizontal negative shift with a mirrored D.
        result.horizontal()[rows, cols - r[:,np.newaxis]] = D
        result.vertical()[rows, cols - (colors*colors-1) + r[:,np.newaxis]] = D[::-1,:]
        

    # Even colors
    else:

        # create double Domino strings
        D0,D1 = create_double_domino(colors)

        # assign D to vertical (w/ positive shift), and D0,D1 to horizonal (w/o shift)
        # Note: a positive horizontal shift == vertical positive shift
        result.vertical()[rows, cols - r[:,np.newaxis]] = D
        result.horizontal()[0::2,:] = D0
        result.horizontal()[1::2,:] = D1

    # Done.
    return result
