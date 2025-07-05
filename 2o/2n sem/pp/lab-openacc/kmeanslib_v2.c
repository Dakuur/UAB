#include <stdio.h>
#include <time.h>
#include <stdlib.h>
#include <sys/time.h>
#include <unistd.h>
#include <limits.h>
#include <string.h>
#include <stdint.h>
#include <inttypes.h>
#include "kmeanslib.h"

/*
 * Read File function
 *
 * mImage->pixels = malloc(mImage->length * sizeof(rgb));
 *
 */
int read_file(char *name, image *mImage)
{
	if ((mImage->fp = fopen(name, "r")) == NULL)
	{
		printf("No fue posible abrir el BMP.\n");
		exit(6);
	}
	if (fseek(mImage->fp, 18, SEEK_SET) != 0)
	{
		printf("Error fseek.\n");
		exit(6);
	}
	if (fread(&mImage->width, sizeof(mImage->width), 1, mImage->fp) != 1)
	{
		printf("Error Writing File.\n");
		exit(6);
	}
	if (fseek(mImage->fp, 22, SEEK_SET) != 0)
	{
		printf("Error fseek.\n");
		exit(6);
	}
	if (fread(&mImage->height, sizeof(mImage->height), 1, mImage->fp) != 1)
	{
		printf("Error Reading H File.\n");
		exit(6);
	}

	mImage->length = (mImage->width * mImage->height);

	mImage->pixels = malloc(mImage->length * sizeof(rgb));

	if (mImage->pixels == NULL)
	{
		printf("Not enough memory.\n");
		return 6;
	}

	if (fseek(mImage->fp, 0, SEEK_SET) != 0)
	{
		printf("Error fseek.\n");
		return 6;
	}

	if (fread(&mImage->header, sizeof(uint8_t), sizeof(mImage->header), mImage->fp) != sizeof(mImage->header))
	{
		printf("Error reading header.\n");
		return 6;
	}

	int i, c;

	for (i = 0; i < mImage->length; i++)
	{
		c = getc(mImage->fp);
		if (c == EOF)
		{
			printf("Error Reading File.\n");
			return 5;
		}
		mImage->pixels[i].b = c;
		c = getc(mImage->fp);
		if (c == EOF)
		{
			printf("Error Reading File.\n");
			return 5;
		}
		mImage->pixels[i].g = c;
		c = getc(mImage->fp);
		if (c == EOF)
		{
			printf("Error Reading File.\n");
			return 5;
		}
		mImage->pixels[i].r = c;
	}

	if (fclose(mImage->fp) != 0)
	{
		printf("Error cerrando fichero.\n");
		return 1;
	}
	return (0);
}

int write_file(char *name, image *mImage, cluster *centroids, uint8_t k)
{
	FILE *fp;
	int i, j;
	uint8_t closest;
	rgb pixel;

	if ((fp = fopen(name, "w")) == NULL)
	{
		printf("Cannot create output file, do you have write permissions for current directory?.\n");
		return 1;
	}
	if (fwrite(mImage->header, sizeof(uint8_t), sizeof(mImage->header), fp) != sizeof(mImage->header))
	{
		printf("Error writing BMP.\n");
		return 1;
	}
	for (i = 0; i < mImage->width; i++)
	{
		for (j = 0; j < mImage->height; j++)
		{
			closest = find_closest_centroid(&mImage->pixels[i * mImage->height + j], centroids, k);
			pixel.r = centroids[closest].r;
			pixel.g = centroids[closest].g;
			pixel.b = centroids[closest].b;

			if (fwrite(&pixel, sizeof(uint8_t), 3, fp) != 3)
			{
				printf("Error writing BMP.\n");
				return 1;
			}
		}
	}
	if (fclose(fp) != 0)
	{
		printf("Error closing file.\n");
		return 1;
	}
	if ((fp = fopen("clusters.txt", "w")) == NULL)
	{
		printf("Clusters.txt cannot be created. Do you have space | owner in current directory?\n");
		return 1;
	}

	fprintf(fp, "Red\tGreen\tBlue\n");

	for (j = 0; j < k; j++)
	{
		fprintf(fp, "%d\t%d\t%d\n", centroids[j].r, centroids[j].g, centroids[j].b);
	}

	if (fclose(fp) != 0)
	{
		printf("Error closing fichero.\n");
		return 1;
	}
}

/*
 * Compute Checksum
 * Checksum is the total sum of the product (r,g,b) of the centroids final value.
 *
 */
uint32_t getChecksum(cluster *centroids, uint8_t k)
{
	uint32_t i, j;
	uint32_t sum = 0;

	for (i = 0; i < k; i++)
	{
		printf("Centroide %u : R[%u]\tG[%u]\tB[%u]\n", i, centroids[i].r, centroids[i].g, centroids[i].b);
		sum += (centroids[i].r * centroids[i].g * centroids[i].b);
	}
	return sum;
}

/*
 * Return the index number of the closest centroid to a given pixel (p)
 * input @param: p 				--> pixel pointer
 * input @param: centroids 		--> cluster array pointer
 * input @param: num_clusters	--> amount of centroids in @param: centroids
 *
 * output: uint8_t --> Index of the closest centroid
 *
 */
uint8_t find_closest_centroid(rgb *p, cluster *centroids, uint8_t num_clusters)
{
	uint32_t min = UINT32_MAX;
	uint32_t dis[num_clusters];
	uint8_t closest = 0, j;
	int16_t diffR, diffG, diffB;

	for (j = 0; j < num_clusters; j++)
	{
		diffR = centroids[j].r - p->r;
		diffG = centroids[j].g - p->g;
		diffB = centroids[j].b - p->b;
		// No sqrt required.
		dis[j] = diffR * diffR + diffG * diffG + diffB * diffB;

		if (dis[j] < min)
		{
			min = dis[j];
			closest = j;
		}
	}
	return closest;
}

/*
 * Main function k-means
 * input @param: K 			--> number of clusters
 * input @param: centroides --> the centroids
 * input @param: num_pixels --> number of total pixels
 * input @param: pixels 	--> pinter to array of rgb (pixels)
 */
void kmeans(uint8_t k, cluster *centroides, uint32_t num_pixels, rgb *pixels)
{
	uint8_t condition, changed, closest;
	uint32_t i, j, random_num;

	printf("STEP 1: K = %d\n", k);
	k = MIN(k, num_pixels);

	// Init centroids
	printf("STEP 2: Init centroids\n");
	for (i = 0; i < k; i++)
	{
		random_num = rand() % num_pixels;
		centroides[i].r = pixels[random_num].r;
		centroides[i].g = pixels[random_num].g;
		centroides[i].b = pixels[random_num].b;
	}

	// K-means iterative procedures start
	printf("STEP 3: Updating centroids\n\n");
	i = 0;
	do
	{
		// Reset centroids
		for (j = 0; j < k; j++)
		{
			centroides[j].media_r = 0;
			centroides[j].media_g = 0;
			centroides[j].media_b = 0;
			centroides[j].num_puntos = 0;
		}

		/*---------------------------------------------------------------------*/
		//							   CODI MODIFICAT
		/*---------------------------------------------------------------------*/

		// Find closest cluster for each pixel

		// Inicialitzem les variables auxiliars que farem servir amb calloc()
		// Tots els valors per cada cluster a 0 (R, G, B i points)
		// printf("Informació del codi modificat:\n");
		// printf("	Declarant variables...\n");

		uint32_t red, green, blue, points = 0;

		// uint8_t *closest_per_pixel = calloc(num_pixels, sizeof(uint8_t));
		uint32_t **pixels_per_cluster = malloc(k * sizeof(uint8_t *));
		for (int i = 0; i < k; i++)
		{
			pixels_per_cluster[i] = calloc(num_pixels, sizeof(uint32_t));
		}
		uint32_t *counters = calloc(k, sizeof(uint32_t));

		uint32_t *current_pixels = malloc(sizeof(uint32_t) * num_pixels);
		uint32_t num_current_pixels = 0;
		uint32_t p = 0;

		// Calculem el clúster més proper per a cada píxel i els pixels per clúster
		// #pragma acc data copyin(pixels[0:num_pixels], centroides[0:k], pixels_per_cluster[0:k][0:num_pixels], counters[0:k]) copyout(pixels_per_cluster[0:k][0:num_pixels], counters[0:k])
		{
			//#pragma acc parallel loop private(closest)
			for (p = 0; p < num_pixels; p++)
			{
				closest = find_closest_centroid(&pixels[p], centroides, k);
				pixels_per_cluster[closest][counters[closest]] = p;
				counters[closest]++;
				// closest_per_pixel[p] = closest;
			}
		}

		/*
		for (int c = 0; c < k; c++)
		{
			printf("	Clúster %d: %d píxels\n", c, counters[c]);
		}
		*/

		// printf("	Inicialitzats %d clústers i %d píxels\n", k, num_pixels);

		// Calculem els valors de les variables auxiliars donats els clústers més propers per a cada píxel
		// #pragma acc data copyin(pixels_per_cluster[0:k][0:num_pixels], pixels[0:num_pixels], centroides[0:k], counters[0:k])
		{
			// #pragma acc parallel loop
			for (int c = 0; c < k; c++) // Per cada clúster obtingut
			{
				uint32_t *current_pixels = pixels_per_cluster[c];
				uint32_t num_current_pixels = counters[c];
				uint32_t red = 0, green = 0, blue = 0;

				// #pragma acc parallel loop reduction(+: red, green, blue) private(p, j)
				for (int j = 0; j < num_current_pixels; j++) // Per cada píxel del clúster
				{
					p = current_pixels[j];
					red += pixels[p].r;
					green += pixels[p].g;
					blue += pixels[p].b;
					// points++; Es pot ometre, ja que ja tenim el nombre de punts per clúster
				}

				// Actualitzem els valors dels centroides (k iteracions, nombre baix i no cal paral·lelitzar)
				centroides[c].media_r = red;
				centroides[c].media_g = green;
				centroides[c].media_b = blue;
				centroides[c].num_puntos = num_current_pixels;
			}
		}

		/*
		// Alliberem la memòria de les variables auxiliars
		free(red);
		free(green);
		free(blue);
		free(points);
		free(closest_per_pixel);

		for(int i = 0; i < k; i++)
		{
			free(pixels_per_cluster[i]);
		}
		free(pixels_per_cluster);
		free(counters);
		printf("	Memòria alliberada\n\n");
		*/

		/*---------------------------------------------------------------------*/
		//							FI CODI MODIFICAT
		/*---------------------------------------------------------------------*/

		// Update centroids & check stop condition
		condition = 0;
		for (j = 0; j < k; j++)
		{
			if (centroides[j].num_puntos == 0)
			{
				continue;
			}

			centroides[j].media_r = centroides[j].media_r / centroides[j].num_puntos;
			centroides[j].media_g = centroides[j].media_g / centroides[j].num_puntos;
			centroides[j].media_b = centroides[j].media_b / centroides[j].num_puntos;
			changed = centroides[j].media_r != centroides[j].r || centroides[j].media_g != centroides[j].g || centroides[j].media_b != centroides[j].b;
			condition = condition || changed;
			centroides[j].r = centroides[j].media_r;
			centroides[j].g = centroides[j].media_g;
			centroides[j].b = centroides[j].media_b;
		}

		i++;
	} while (condition);
	printf("Number of K-Means iterations: %d\n\n", i);
}
