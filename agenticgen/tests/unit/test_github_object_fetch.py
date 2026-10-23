"""Remote standalone checks of scoped Git tree verification; never access network."""
import copy
import importlib.util
from pathlib import Path
import unittest

here=Path(__file__).resolve()
source=here.with_name('fetch_fixed_github_api.py')
if not source.exists():source=here.parents[2]/'scripts/fetch_fixed_github_api.py'
spec=importlib.util.spec_from_file_location('fixed_fetch',source)
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)


class ScopedTrees(unittest.TestCase):
    def setUp(self):
        self.original=module.PREFIXES;module.PREFIXES=('agenticgen/',)
        self.blob=module.object_id('blob',b'code')
        self.leaf=[{'name':'code.py','path':'code.py','mode':'100644','type':'blob','sha':self.blob}]
        self.subtree=module.object_id('tree',module.tree_bytes(self.leaf))
        self.outside='a'*40
        self.root_entries=[{'name':'agenticgen','path':'agenticgen','mode':'040000','type':'tree','sha':self.subtree},
                           {'name':'unrelated','path':'unrelated','mode':'040000','type':'tree','sha':self.outside}]
        self.root=module.object_id('tree',module.tree_bytes(self.root_entries))
        self.docs={'trees/'+self.root:{'sha':self.root,'truncated':False,'tree':self.root_entries},
                   'trees/'+self.subtree+'?recursive=1':{'sha':self.subtree,'truncated':False,'tree':self.leaf}}
        self.calls=[];self.writes=[]
    def tearDown(self):module.PREFIXES=self.original
    def api(self,key):self.calls.append(key);return copy.deepcopy(self.docs[key])
    def put(self,kind,raw,expected):
        self.assertEqual(module.object_id(kind,raw),expected);self.writes.append(expected)
    def run_fetch(self,known=None):return module.scoped_trees(self.api,self.root,{self.outside} if known is None else known,self.put)
    def test_only_required_subtree_recurses(self):
        blobs,requests=self.run_fetch()
        self.assertEqual(blobs,{self.blob});self.assertEqual(requests,self.calls)
        self.assertNotIn('trees/'+self.root+'?recursive=1',requests)
        self.assertNotIn('trees/'+self.outside,requests)
        self.assertEqual(set(self.writes),{self.root,self.subtree})
    def test_cached_trees_still_bind_blob_ids(self):
        blobs,_=self.run_fetch({self.outside,self.root,self.subtree})
        self.assertEqual(blobs,{self.blob});self.assertEqual(self.writes,[])
    def test_truncated_subtree_rejected(self):
        self.docs['trees/'+self.subtree+'?recursive=1']['truncated']=True
        with self.assertRaisesRegex(ValueError,'Incomplete'):self.run_fetch()
    def test_wrong_root_identity_rejected(self):
        self.docs['trees/'+self.root]['sha']='f'*40
        with self.assertRaisesRegex(ValueError,'Incomplete'):self.run_fetch()
    def test_modified_blob_reference_rejected(self):
        self.docs['trees/'+self.subtree+'?recursive=1']['tree'][0]['sha']='b'*40
        with self.assertRaisesRegex(ValueError,'digest'):self.run_fetch()
    def test_duplicate_paths_rejected(self):
        self.docs['trees/'+self.subtree+'?recursive=1']['tree'].append(copy.deepcopy(self.leaf[0]))
        with self.assertRaisesRegex(ValueError,'duplicate'):self.run_fetch()
    def test_escape_path_rejected(self):
        self.docs['trees/'+self.subtree+'?recursive=1']['tree'][0]['path']='../code.py'
        with self.assertRaisesRegex(ValueError,'Unsafe'):self.run_fetch()
    def test_unknown_unrelated_tree_rejected(self):
        with self.assertRaisesRegex(ValueError,'outside deployment'):self.run_fetch(set())
    def test_required_subtree_missing_rejected(self):
        module.PREFIXES=('absent/',)
        with self.assertRaisesRegex(ValueError,'subtree missing'):self.run_fetch({self.subtree,self.outside})


if __name__=='__main__':unittest.main()
