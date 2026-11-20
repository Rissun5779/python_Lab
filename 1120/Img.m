x = imread('img1.jpg')

figure
imshow(x)

figure
imshow(x(:,:,1))
#1
mask = [0 -1 0; -1 4 -1; 0 -1 0]

y(:,:,1) = conv2(x(:,:,1), mask)
y(:,:,2) = conv2(x(:,:,2), mask)
y(:,:,3) = conv2(x(:,:,3), mask)

y = uint8(abs(y+128))

figure
imshow(uint8(y))
#2
mask = [0 -1 0; -1 6 -1; 0 -1 0]

y(:,:,1) = conv2(x(:,:,1), mask)
y(:,:,2) = conv2(x(:,:,2), mask)
y(:,:,3) = conv2(x(:,:,3), mask)

y = uint8(abs(y))

figure
imshow(uint8(y))
#3
mask = [-2 -1 0; -1 0 1; 0 1 2]

y(:,:,1) = conv2(x(:,:,1), mask)
y(:,:,2) = conv2(x(:,:,2), mask)
y(:,:,3) = conv2(x(:,:,3), mask)

y = uint8(abs(y+128))

figure
imshow(uint8(y))
