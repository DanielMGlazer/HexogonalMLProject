#%%
#Packages
import collections
import math
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont, ImageColor
import numpy as np 
from scipy.stats import norm
from scipy import ndimage

#%%
#Defining functions of hexogonal grid
Point = collections.namedtuple("Point", ["x", "y"])


_Hex = collections.namedtuple("Hex", ["q", "r", "s"])

def Hex(q, r, s): #creates the hexagon datatype 
    assert not (round(q + r + s) != 0), "q + r + s must be 0"
    return _Hex(q, r, s)

def eq(a,b): #Returns wether or not two hexs are equal
    return a.q==b.q and a.r==b.r and a.s==b.s

#Vector operators for hexagons
def hex_add(a, b): 
    return Hex(a.q + b.q, a.r + b.r, a.s + b.s)

def hex_subtract(a, b):
    return Hex(a.q - b.q, a.r - b.r, a.s - b.s)

def hex_scale(a, k:int):
    return Hex(a.q * k, a.r * k, a.s * k)

def hex_rotate_left(a):
    return Hex(-a.s, -a.q, -a.r)

def hex_rotate_right(a):
    return Hex(-a.r, -a.s, -a.q)

def hex_flip_q(a):
    return Hex(-a.q,a.r,a.q-a.r)

def hex_flip_r(a):
    return Hex(a.q,-a.r,-a.q+a.r)

def hex_length(hex): #length of the line from 0,0 to the hexagon
    return (abs(hex.q) + abs(hex.r) + abs(hex.s)) // 2

def hex_distance(a, b): #distance between two hexagons
    return hex_length(hex_subtract(a, b))

#Functions to move directionly in the lattice or aquire neibhbors
hex_directions = [Hex(1, 0, -1), Hex(1, -1, 0), Hex(0, -1, 1), Hex(-1, 0, 1), Hex(-1, 1, 0), Hex(0, 1, -1)]
def hex_direction(direction):
    return hex_directions[direction]

def hex_neighbor(hex, direction):
    return hex_add(hex, hex_direction(direction))


# Datatypes for pointy top or flattop lattices
Orientation = collections.namedtuple("Orientation", ["f0", "f1", "f2", "f3", "b0", "b1", "b2", "b3", "start_angle"])

Layout = collections.namedtuple("Layout", ["orientation", "size", "origin"])

layout_pointy = Orientation(math.sqrt(3.0), math.sqrt(3.0) / 2.0, 0.0, 3.0 / 2.0, math.sqrt(3.0) / 3.0, -1.0 / 3.0, 0.0, 2.0 / 3.0, 0.5)
layout_flat = Orientation(3.0 / 2.0, 0.0, math.sqrt(3.0) / 2.0, math.sqrt(3.0), 2.0 / 3.0, 0.0, -1.0 / 3.0, math.sqrt(3.0) / 3.0, 0.0)


#Functions for locating points in xy coordinates
def hex_to_pixel(layout, h): #returns the x,y coordinate of the center of the hexagon
    M = layout.orientation
    x = (M.f0*h.q+M.f1*h.r)*layout.size.x
    y = (M.f2*h.q+M.f3*h.r)*layout.size.y
    return Point(x + layout.origin.x , y + layout.origin.y)


def hex_corner_offset(layout, corner): #Returns how far off each corner is from the center point
    M = layout.orientation
    size = layout.size
    angle = 2.0 * math.pi * (M.start_angle - corner) / 6.0
    return Point(size.x * math.cos(angle), size.y * math.sin(angle))

def polygon_corners(layout, h): #Creates an array of corners by applying the corner offset method to the center six times
    corners = []
    center = hex_to_pixel(layout, h)
    for i in range(0, 6):
        offset = hex_corner_offset(layout, i)
        corners.append(Point(center.x + offset.x, center.y + offset.y))
    return corners

def to_tuple(corners): #turns corners into an array of tuples 
    tuple_corners=[]
    for p in corners:
        tuple_corners.append((p.x,p.y))
    return tuple_corners

def to_array(corners): # turns corners into an array of arrays for coordinates
    array_corners=[]
    for p in corners:
        array_corners.append([p.x,p.y])
    return array_corners


# Creates rectangular map of hexagons for flat top, origin is in top left
def rect_map(map_height,map_width):
    map=[]
    for q in range(map_height):
        q_offset=math.floor(q/2)
        for r in range(-q_offset,map_width-q_offset):
            #map.append(Hex(q,r-map_height//2,-(q)-(r-map_height//2)))
            map.append(Hex(q,r,-q-r))
    return map


#%%
#Functions for drawing hexagons
def plot_hex_grid(h,line_width,color,draw):
    corners=to_tuple(polygon_corners(layout,h))
    #corners.append[corners[0]]
    draw.line(corners[:4],fill=color,width=line_width,joint='curve') #only draws half of each hexagon

def plot_hex_grid_text(h,draw):   #writes the q,r coordinates in the middle of each hexagon
    corners=to_tuple(polygon_corners(layout,h))
    draw.polygon(corners,outline='white')
    shift=d.textsize(f"({h.q},{h.r})")
    draw.text((hex_to_pixel(layout,h).x-shift[0]//2,hex_to_pixel(layout,h).y-shift[1]//2),f"({h.q},{h.r})",fill='white')

def solid_circ(p,radius,color,draw):
    p1=(p[0]-radius,p[1]-radius)
    p2=(p[0]+radius,p[1]+radius)
    draw.ellipse([p1,p2],fill=color)

def draw_circ(p,radius,color,draw):
    p1=(p[0]-radius,p[1]-radius)
    p2=(p[0]+radius,p[1]+radius)
    draw.ellipse([p1,p2],outline=color)


def Gauss_circ(p,radius,color,draw):
    _scale=0.5*radius
    for i in range(0,2*radius):
        color_scaling=math.sqrt(2*3.1415*_scale)*240*norm.pdf(i,scale=_scale)
        color_circ=f"hsv({max(color_scaling,.01)},100%,50%)"
        draw_circ(p,i,color_circ,draw)


#Function for drawing gaussian dots pixel by pixel
Gaussian= lambda x,scl,amp: amp*math.e**(-0.5*(x/scl)**2)
def Gauss_circ_pixel(p,radius):
    p[0]=int(p[0])
    p[1]=int(p[1])
    #radius is a given param that represents the standard deviation of the gaussian
    #plot_rad is some multiple of the radius given, it is the radius of the circle to be plotted
    _scale=radius
    plot_rad=4*radius
    color_amp=100
    color_scaling=lambda x: Gaussian(x,_scale,color_amp)

    for x in range(max(p[0]-plot_rad,0),min(p[0]+plot_rad,image_size[0])):
        y_max=math.floor(math.sqrt(plot_rad**2-(x-p[0])**2))
        for y in range(max(p[1]-y_max,0),min(p[1]+y_max,image_size[1])):
            dist_vec=np.array((x-p[0],y-p[1]))
            distance=np.linalg.norm(dist_vec)
            blob_values_array[x][y]+=color_scaling(distance)
            
def gauss_to_color():
    for i in range(image_size[0]):
        for j in range(image_size[1]):
            #rgbcolor=ImageColor.getrgb(f"hsl({math.floor(blob_values_array[i][j])},100%,50%)")
            rgbcolor=ImageColor.getrgb(f"hsl(46,{min(math.floor(blob_values_array[i][j]),100)}%,50%)")
            pixel_array[i][j][0]=rgbcolor[0]
            pixel_array[i][j][1]=rgbcolor[1]
            pixel_array[i][j][2]=rgbcolor[2]



def plot_hex_dots(h,radius,color,draw,_type):
    assert _type=="Circle" or _type=="Gaussian" or _type=="Gaussian_pixel"
    corners=to_array(polygon_corners(layout,h))
    #corners.append(hex_to_pixel(layout,h)) #Drawing center dot as well. Not needed 
    for p in corners[:2]:
        if _type == 'Circle':
            solid_circ(p,radius,color,draw)
        if _type == "Gaussian":
            Gauss_circ(p,radius,color,draw)
        if _type=="Gaussian_pixel":
            Gauss_circ_pixel(p,radius)

#%%
#Noise creation functions
def make_blurred_image(image,s):
    image_n = ndimage.filters.gaussian_filter(image, s/20)
    return image_n

def make_noisy_image(image,l):
    vals = len(np.unique(image))
    vals = (l/75) ** np.ceil(np.log2(vals))
    image_n_filt = np.random.poisson(image* vals) / float(vals)
    return image_n_filt


#%%
#Drawing Hexogonal grid
orange=(202,163,24)
image_size=(512,512)
background_color=ImageColor.getrgb("hsl(46,0%,50%)")
im=Image.new("RGB",image_size,color=background_color)
pixel_array=np.array(im)
blob_values_array=np.zeros((image_size[0],image_size[1]))
d=ImageDraw.Draw(im)
#origin=Point(image_size[0]//2,image_size[1]//2) #middle of the imgage
origin=Point(0,0) #Top left corner
size=Point(100,100)
dot_size=math.floor(0.3*size[0])
line_width=math.floor(.3*size[0])
structure_color=orange
layout= Layout(layout_flat,size,origin)
map=rect_map(4,4)

#solid_circ(origin,300,"hsv(240,100%,50%)",d)
#Gauss_circ_pixel((0,0),40)
for h in map:
    plot_hex_dots(h,dot_size,structure_color,d,"Gaussian_pixel")
gauss_to_color()
im=Image.fromarray(pixel_array)

fig = plt.figure(figsize = (10, 10))
ax = plt.subplot(1,1,1)
ax.imshow(im)
ax.axis('off')
#im.save("Hex_lat_Gauss_blobs_biggeratoms.png")


#%% 
im1=make_noisy_image(im,50)
fig = plt.figure(figsize = (10, 10))
ax = plt.subplot(1,1,1)
ax.imshow(im)
ax.axis('off')
#%%
#Test functions
def complain(name):
    print("FAIL {0}".format(name))
def equal_hex(name, a, b):
    if not (a.q == b.q and a.s == b.s and a.r == b.r):
        complain(name)
def equal_int(name, a, b):
    if not (a == b):
        complain(name)
def equal_hex_array(name, a, b):
    equal_int(name, len(a), len(b))
    for i in range(0, len(a)):
        equal_hex(name, a[i], b[i])
def test_hex_arithmetic():
    equal_hex("hex_add", Hex(4, -10, 6), hex_add(Hex(1, -3, 2), Hex(3, -7, 4)))
    equal_hex("hex_subtract", Hex(-2, 4, -2), hex_subtract(Hex(1, -3, 2), Hex(3, -7, 4)))
def test_hex_direction():
    equal_hex("hex_direction", Hex(0, -1, 1), hex_direction(2))
def test_hex_neighbor():
    equal_hex("hex_neighbor", Hex(1, -3, 2), hex_neighbor(Hex(1, -2, 1), 2))
def test_hex_distance():
    equal_int("hex_distance", 7, hex_distance(Hex(3, -7, 4), Hex(0, 0, 0)))
def test_all():
    test_hex_arithmetic()
    test_hex_direction()
    test_hex_neighbor()
    test_hex_distance()
test_all()



#%%
x=np.linspace(-2,2)
plt.plot(x,norm.pdf(x,scale=.5))
plt.show

#%%
