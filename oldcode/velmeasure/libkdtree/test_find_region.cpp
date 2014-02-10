// Thanks to James Remillard
//
#include <cstdio>
#include <kdtree++/kdtree.hpp>
#include <vector>
#include <map>
#include <set>

using namespace std;

double rlow = 0.1;
double rhigh = 0.2;

struct kdtreeNode
{
 typedef float value_type;

 float xyz[3];
 size_t index;

 value_type operator[](size_t n) const
 {
   return xyz[n];
 }

};

int main(int argc,char *argv[])
{
    vector<kdtreeNode> pts;
    
    typedef KDTree::KDTree<3,kdtreeNode> treeType;
    
    treeType tree;
    srand (time(NULL));
    
    // make random 3d points
    for ( size_t n = 0; n < 10000; ++n)
    {
        kdtreeNode node;
        node.xyz[0] = double(rand())/RAND_MAX;
        node.xyz[1] = double(rand())/RAND_MAX;
        node.xyz[2] = double(rand())/RAND_MAX;
        node.index = n;
        
        tree.insert( node);
        pts.push_back( node);
    }
    
    kdtreeNode node;
    treeType::_Region_ * region = new treeType::_Region_(node);
    
    //kdtreeNode refnode;
    /*refnode.xyz[0] = 0.1;
    refnode.xyz[1] = 0.1;
    refnode.xyz[2] = 0.1;
    region.set_low_bound(refnode, 0);
    region.set_low_bound(refnode, 1);
    region.set_low_bound(refnode, 2);
    refnode.xyz[0] = 0.2;
    refnode.xyz[1] = 0.2;
    refnode.xyz[2] = 0.2;
    region.set_high_bound(refnode, 0);
    region.set_high_bound(refnode, 1);
    region.set_high_bound(refnode, 2);*/
    region->_M_low_bounds[0] = rlow;
    region->_M_low_bounds[1] = rlow;
    region->_M_low_bounds[2] = rlow;
    region->_M_high_bounds[0] = rhigh;
    region->_M_high_bounds[1] = rhigh;
    region->_M_high_bounds[2] = rhigh;
    
    set<size_t> correctCloseList;
    // now do the same with the kdtree.
    vector<kdtreeNode> howClose;
    tree.find_within_range(*region, back_insert_iterator<vector<kdtreeNode> >(howClose));
    
    for ( size_t i= 0; i < pts.size(); ++i)
    {
        if (    pts[i].xyz[0] < rhigh &&  pts[i].xyz[0] > rlow &&
                pts[i].xyz[1] < rhigh &&  pts[i].xyz[1] > rlow &&
                pts[i].xyz[2] < rhigh &&  pts[i].xyz[2] > rlow)
            correctCloseList.insert( i );
    }
    
   for ( size_t i = 0; i < howClose.size(); ++i)
   {
     printf("%f %f %f\n",
            howClose[i].xyz[0],
            howClose[i].xyz[1],
            howClose[i].xyz[2]);
     set<size_t>::iterator hit = correctCloseList.find( howClose[i].index);

     if ( hit != correctCloseList.end())
     {
       correctCloseList.erase(hit);
     }
     else
     {
       // point that is too far away - fail!
       assert(false);
       printf("fail, extra points.\n");
     }

   }

   // fail, not all of the close enough points returned.
   assert( correctCloseList.size() == 0);
   if ( correctCloseList.size() > 0)
   {
     printf("fail, missing points.\n");
   }
}

