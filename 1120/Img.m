x = imread('img1.jpg')

figure
imshow(x)

figure
imshow(x(:,:,1))
#1
mask = [-1 -1 -1; -1 8 -1; -1 -1 -1]

y(:,:,1) = conv2(x(:,:,1), mask)
y(:,:,2) = conv2(x(:,:,2), mask)
y(:,:,3) = conv2(x(:,:,3), mask)

y = uint8(abs(y+128))

figure
imshow(uint8(y))
#2
mask = [0 -1 0; -1 5 -1; 0 -1 0]

y(:,:,1) = conv2(x(:,:,1), mask)
y(:,:,2) = conv2(x(:,:,2), mask)
y(:,:,3) = conv2(x(:,:,3), mask)

y = uint8(abs(y+128))

figure
imshow(uint8(y))
#3
mask = [1/9 1/9 1/9; 1/9 1/9 1/9; 1/9 1/9 1/9]

y(:,:,1) = conv2(x(:,:,1), mask)
y(:,:,2) = conv2(x(:,:,2), mask)
y(:,:,3) = conv2(x(:,:,3), mask)

y = uint8(abs(y+128))

figure
imshow(uint8(y))
