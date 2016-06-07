#include <iostream>

#include <boost/thread/thread.hpp>
#include <pcl/common/common_headers.h>
#include <pcl/features/normal_3d.h>
#include <pcl/io/pcd_io.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/console/parse.h>

#include <pcl/ModelCoefficients.h>
#include <pcl/point_types.h>
#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/segmentation/extract_clusters.h>

#include <pcl/filters/passthrough.h>
#include <pcl/filters/radius_outlier_removal.h>
#include <pcl/segmentation/region_growing.h>
#include <pcl/segmentation/region_growing_rgb.h>
#include <pcl/filters/statistical_outlier_removal.h>

using namespace pcl;
using namespace std;

class clSeg
{
private:

public:
  clSeg()
  {
  }
  void planeSegmentation(pcl::PointCloud<PointXYZRGB>::Ptr cloud_input, PointCloud<PointXYZRGB>::Ptr segmented)
  {
    PointCloud<PointXYZRGB>::Ptr cloud_filtered (new PointCloud<PointXYZRGB>);
    PointCloud<PointXYZRGB>::Ptr cloud_inliers (new PointCloud<PointXYZRGB>);
    ModelCoefficients::Ptr coefficients (new ModelCoefficients ());
    PointIndices::Ptr inliers (new PointIndices ());

    SACSegmentation<PointXYZRGB> seg;
    seg.setOptimizeCoefficients (true);
    seg.setModelType (SACMODEL_PLANE);
    seg.setMethodType (SAC_RANSAC);
    seg.setMaxIterations (1000);
    seg.setDistanceThreshold (0.01);

    *cloud_filtered = *cloud_input;

    int i = 0, nr_points = cloud_filtered->points.size();
    while(cloud_filtered->points.size() > 0.3*nr_points)
      {
	// Segment the largest planar component from the cloud
	seg.setInputCloud (cloud_filtered);
	seg.segment (*inliers, *coefficients);
	if (inliers->indices.size () == 0)
	  {
	    cerr << "Could not estimate a planar model for the given dataset." << std::endl;
	  }

	ExtractIndices<PointXYZRGB> extract(true);
	// Extract the inliers
	extract.setInputCloud(cloud_filtered);
	extract.setIndices(inliers);
	extract.setNegative (false);
	extract.filter (*cloud_inliers);
	extract.setNegative(true);
	extract.filter(*cloud_filtered);
	i++;
      }
    *segmented = *cloud_filtered;
  }

  void regionGrowingSegmentation(PointCloud<PointXYZRGB>::Ptr cloud_input, vector<PointIndices>& indices)
  {
    search::Search<PointXYZRGB>::Ptr tree = boost::shared_ptr<search::Search<PointXYZRGB> > (new search::KdTree<PointXYZRGB>);
    PointCloud <Normal>::Ptr normals (new PointCloud <Normal>);
    NormalEstimation<PointXYZRGB, Normal> normal_estimator;
    normal_estimator.setSearchMethod (tree);
    normal_estimator.setInputCloud (cloud_input);
    normal_estimator.setKSearch (50);
    normal_estimator.compute (*normals);

    IndicesPtr pass_indices (new std::vector <int>);
    PassThrough<PointXYZRGB> pass;
    pass.setInputCloud (cloud_input);
    pass.setFilterFieldName ("z");
    pass.setFilterLimits (-20.0, 20.0);
    pass.filter (*pass_indices);

    RegionGrowing<PointXYZRGB, Normal> reg;
    reg.setMinClusterSize (600);
    reg.setMaxClusterSize (1000000);
    reg.setSearchMethod (tree);
    reg.setNumberOfNeighbours (30);
    reg.setInputCloud (cloud_input);
    reg.setIndices (pass_indices);
    reg.setInputNormals (normals);
    reg.setSmoothnessThreshold (10.0 / 180.0 * M_PI);
    reg.setCurvatureThreshold (1.0);

    std::vector <PointIndices> cluster_indices;
    reg.extract (cluster_indices);

    indices.resize(cluster_indices.size());
    indices = cluster_indices;
  }

  void clusterExtractionRGB(PointCloud<PointXYZRGB>::Ptr cloud_input, vector<PointIndices>& indices)
  {
    PointCloud<PointXYZRGB>::Ptr cloud_filtered (new PointCloud<PointXYZRGB>);
    search::KdTree<PointXYZRGB>::Ptr tree (new search::KdTree<PointXYZRGB>);
    IndicesPtr pass_indices (new std::vector <int>);
    vector<PointIndices> cluster_indices;
    tree->setInputCloud (cloud_input);

    PassThrough<pcl::PointXYZRGB> pass;
    pass.setInputCloud (cloud_input);
    pass.setFilterFieldName ("z");
    pass.setFilterLimits (0.0, 10.0);
    pass.filter (*pass_indices);
    pass.setFilterFieldName ("x");
    pass.setFilterLimits (0.0, 10.0);
    pass.filter (*pass_indices);
    pass.setFilterFieldName ("y");
    pass.setFilterLimits (0.0, 10.0);
    pass.filter (*pass_indices);

    EuclideanClusterExtraction<PointXYZRGB> ec;
    ec.setClusterTolerance (0.02); // 2cm
    ec.setMinClusterSize (100);
    ec.setMaxClusterSize (250000);
    ec.setSearchMethod (tree);
    ec.setInputCloud (cloud_input);
    ec.setIndices(pass_indices);
    ec.extract (cluster_indices);

    indices.resize(cluster_indices.size());
    indices = cluster_indices;
  }

  void colorRegionGrowingSegmentation(PointCloud<PointXYZRGB>::Ptr cloud_input, vector<PointIndices>& indices)
  {
    search::Search <pcl::PointXYZRGB>::Ptr tree = boost::shared_ptr<pcl::search::Search<pcl::PointXYZRGB> > (new pcl::search::KdTree<pcl::PointXYZRGB>);
    IndicesPtr pass_indices (new std::vector <int>);
    PassThrough<pcl::PointXYZRGB> pass;
    pass.setInputCloud (cloud_input);
    pass.setFilterFieldName ("z");
    pass.setFilterLimits (0.0, 10.0);
    pass.filter (*pass_indices);

    RegionGrowingRGB<pcl::PointXYZRGB> reg;
    reg.setInputCloud (cloud_input);
    reg.setIndices (pass_indices);
    reg.setSearchMethod (tree);
    reg.setDistanceThreshold (10);
    reg.setPointColorThreshold (6);
    reg.setRegionColorThreshold (5);
    reg.setMinClusterSize (600);

    vector <pcl::PointIndices> cluster_indices;
    reg.extract (cluster_indices);

    indices.resize(cluster_indices.size());
    indices = cluster_indices;
    cout << indices.size() << endl;
  }
  void downsampleCloud(pcl::PCLPointCloud2::Ptr cloud)
  {
    pcl::VoxelGrid<pcl::PCLPointCloud2> vox;
    vox.setInputCloud (cloud);
    vox.setLeafSize (0.01f, 0.01f, 0.01f);
    vox.filter (*cloud);
  }
  void removeOutliers(PointCloud<PointXYZRGB>::Ptr cloud_input)
  {
    pcl::StatisticalOutlierRemoval<pcl::PointXYZRGB> sor;
    sor.setInputCloud (cloud_input);
    sor.setMeanK (50);
    sor.setStddevMulThresh (1.0);
    sor.filter (*cloud_input);
    cout << "tst";
  }
};

int main (int argc, char** argv)
{
  clSeg cl;
  PCLPointCloud2::Ptr pcd (new pcl::PCLPointCloud2);
  PointCloud<PointXYZ>::Ptr cloud (new PointCloud<PointXYZ>), cloud_inliers (new PointCloud<PointXYZ>);
  PointCloud<PointXYZRGB>::Ptr cloud_rgb (new PointCloud<PointXYZRGB>), cloud_inliers2 (new PointCloud<PointXYZRGB>), cloud_seg (new PointCloud<PointXYZRGB>);
  vector<PointIndices> cluster_indices;


  PCDReader reader;
  reader.read ("../clouds/table_scene_lms400.pcd", *pcd);
  //cl.downsampleCloud(pcd);

  fromPCLPointCloud2 (*pcd, *cloud_rgb);
  cl.planeSegmentation(cloud_rgb, cloud_seg);
  cl.removeOutliers(cloud_seg);
  cl.regionGrowingSegmentation(cloud_seg, cluster_indices);
  //cl.colorRegionGrowingSegmentation(cloud_seg, cluster_indices);
  //cl.clusterExtractionRGB(cloud_seg, cluster_indices);

  visualization::PCLVisualizer viewer("3D Viewer");
  viewer.setBackgroundColor (0, 0, 0);
  std::stringstream ss;
  //viewer.addPointCloud<pcl::PointXYZRGB> (cloud_seg, "sample_cloud");
  for(int i = 0; i< cluster_indices.size(); i++)
    {
      PointIndices::Ptr inliers(new PointIndices);
      inliers->indices = cluster_indices[i].indices;

      pcl::ExtractIndices<pcl::PointXYZRGB> extract(true);
      extract.setInputCloud(cloud_seg);
      extract.setIndices(inliers);
      extract.setNegative (false);
      extract.filter (*cloud_inliers2);

      ss << "cloud_cluster_" << i;
      visualization::PointCloudColorHandlerCustom<pcl::PointXYZRGB> single_color(cloud_inliers2, (rand() % 255) +1, (rand() % 255) +1, (rand() % 255) +1);
      viewer.addPointCloud<pcl::PointXYZRGB> (cloud_inliers2, single_color, ss.str());
    }

  //visualization
  //viewer.addPointCloud<pcl::PointXYZRGB> (cloud_seg, "sample_cloud");

  viewer.setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, ss.str());
  //viewer.addCoordinateSystem (1.0);
  //viewer.initCameraParameters ();

  while (!viewer.wasStopped ())
  {
    viewer.spinOnce (100);
  }

  return 0;
}
