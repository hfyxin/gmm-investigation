mean = [0,0,0];
abc = [1.5, 1.5, 3];

[X,Y,Z] = ellipsoid(mean(1), mean(2), mean(3), abc(1), abc(2), abc(3));

s = surf(X,Y,Z);
hold on
s1 = surf(X,Y,Z);
axis equal
hold on
direction = [1 0 0];
rotate(s1,direction,45)
xlabel('x')
ylabel('y')
zlabel('z')