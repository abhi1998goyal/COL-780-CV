class KEYPOINT:
    def __init__(self, i=0, j=0, y=0.0, x=0.0, octave=1, angle=0.0, sigma=0.0, DoG=0):
        self.i = i
        self.j = j
        self.y = y
        self.x = x
        self.octave = octave
        self.angle = angle
        self.sigma = sigma
        self.DoG = DoG