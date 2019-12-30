x = linspace(0, 2*pi, 50);
y1 = sin(x);
y2 = cos(x);
e1 = rand(size(y1))*.5+.5;
e2 = [.25 .5];

ax(1) = subplot(2,2,1);
[l,p] = boundedline(x, y1, e1, '-b*', x, y2, e2, '--ro');
outlinebounds(l,p);
title('Opaque bounds, with outline');
axis tight;


% For our second axis, we use the same 2 lines, and this time assign
% x-varying bounds to both lines.  Rather than using the LineSpec syntax,
% this  example uses the default color order to assign the colors of the
% lines and patches.  I also turn on the |'alpha'| option, which renders
% the patch wit partial transparency.

ax(2) = subplot(2,2,2);
boundedline(x, [y1;y2], rand(length(y1),2,2)*.5+.5, 'alpha');
title('Transparent bounds');
axis tight;


% The bounds can also be assigned to a horizontal orientation, for a case
% where the x-axis represents the dependent variable.  In this case, the
% scalar error bound value applies to both lines and both sides of the
% lines.

ax(3) = subplot(2,2,3);
boundedline([y1;y2], x, e1(1), 'orientation', 'horiz')
title('Horizontal bounds');
axis tight;

% Rather than use a LineSpec or the default color order, a colormap array
% can be used to assign colors.  In this case, increasingly-narrower bounds
% are added on top of the same line.

ax(4) = subplot(2,2,4);
boundedline(x, repmat(y1, 4,1), permute(0.5:-0.1:0.2, [3 1 2]), ...
    'cmap', cool(4), ...
    'transparency', 0.5);
title('Multiple bounds using colormap');

set(ax([1 2 4]), 'xlim', [0 2*pi]);
set(ax(3), 'ylim', [0 2*pi]);
axis tight;

%
% If you plot a line with one or more NaNs in either the |x| or |y| vector,
% the NaN location is rendered as a missing marker with a gap in the line.
% However, the |patch| command does not handle NaNs gracefully; it simply
% fails to show the patch at all if any of the coordinates include NaNs.
%
% Because of this, the expected behavior of the patch part of boundedline
% when confronted with a NaN in either the bounds array (|b|) or the
% x/y-coordinates of the line (which are used to calculate the patch
% coordinates) is ambiguous.  I offer a few options.  
%
% Before I demonstrate the options, I'll create a dataset that has a few
% different types of gaps:

x = linspace(0, 2*pi, 50);
y = sin(x);
b = [ones(size(y))*0.2; rand(size(y))*.5+.5]';

y(10)   = NaN;  % NaN in the line but not bounds
b(20,1) = NaN;  % NaN in lower bound but not line
b(30,2) = NaN;  % NaN in upper bound but not line
b(40,:) = NaN;  % NaN in both sides of bound but not line

% Here's what that looks like in an errorbar plot.

figure;
he = errorbar(x,y,b(:,1), b(:,2), '-bo');


line([x([10 20 30 40]); x([10 20 30 40])], [ones(1,4)*-2;ones(1,4)*2], ...
    'color', ones(1,3)*0.5, 'linestyle', ':');
text(x(10), sin(x(10))-0.2, {'\uparrow','Line','gap'}, 'vert', 'top', 'horiz', 'center');
text(x(20), sin(x(20))-0.2, {'\uparrow','Lower','bound','gap'}, 'vert', 'top', 'horiz', 'center');
text(x(30), sin(x(30))-0.2, {'\uparrow','Upper','bound','gap'}, 'vert', 'top', 'horiz', 'center');
text(x(40), sin(x(40))-0.2, {'\uparrow','Two-sided','bound','gap'}, 'vert', 'top', 'horiz', 'center');

axis tight equal;
 
% The default method for dealing with NaNs in boundedline is to leave the
% gap in the line, but smooth over the gap in the bounds based on the
% neighboring points.  This option can be nice if you only have one or two
% missing points, and you're not interested in emphasizing those gaps in
% your plot:

delete(he);
[hl,hp] = boundedline(x,y,b,'-bo', 'nan', 'fill');
ho = outlinebounds(hl,hp);
set(ho, 'linestyle', ':', 'color', 'r', 'marker', '.');

% I've added bounds outlines in a contrasting color so you can see how I'm
% handling individual points.
%
% The second option leaves a full gap in the patch for any NaN.  I
% considered allowing one-sided gaps, but couldn't think of a good way to
% distinguish a gap from a zero-valued bound.  I'm open to suggestions if
% you have any (email me).

delete([hl hp ho]);
[hl,hp] = boundedline(x,y,b,'-bo', 'nan', 'gap');
ho = outlinebounds(hl,hp);
set(ho, 'linestyle', ':', 'color', 'r', 'marker', '.');


% The final option removes points from the patch that are NaNs.  The visual
% result is very similar to the fill option, but the missing points are
% apparent if you plot the bounds outlines.

delete([hl hp ho]);
[hl,hp] = boundedline(x,y,b,'-bo', 'nan', 'remove');
ho = outlinebounds(hl,hp);
set(ho, 'linestyle', ':', 'color', 'r', 'marker', '.');
