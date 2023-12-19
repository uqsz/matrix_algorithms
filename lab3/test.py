import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Przykładowe dane - zakres pikseli
pixel_range_x = [0, 100]
pixel_range_y = [0, 80]

# Tworzymy subplot
fig, ax = plt.subplots()

# Rysujemy prostokąty
rectangles = [
    patches.Rectangle((10, 10), 20, 30, linewidth=1, edgecolor='r', facecolor='none'),
    patches.Rectangle((40, 20), 15, 25, linewidth=1, edgecolor='g', facecolor='none'),
    patches.Rectangle((70, 40), 25, 20, linewidth=1, edgecolor='b', facecolor='none')
]

# Dodajemy prostokąty do subplotu
for rect in rectangles:
    ax.add_patch(rect)

# Ustawiamy zakresy osi
ax.set_xlim(pixel_range_x)
ax.set_ylim(pixel_range_y)

# Dodajemy etykiety
ax.set_xlabel('Piksele X')
ax.set_ylabel('Piksele Y')
ax.set_title('Prostokąty na wspólnej planszy')

# Wyświetlamy wykres
plt.show()