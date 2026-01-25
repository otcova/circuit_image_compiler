#[derive(Default)]
pub struct UnionFind {
    // Invariant: parents[i] <= i
    // Invariant: len <= u32::MAX + 1
    parents: Vec<u32>,
}

impl Clone for UnionFind {
    fn clone(&self) -> Self {
        Self {
            parents: self.parents.clone(),
        }
    }
    fn clone_from(&mut self, other: &Self) {
        self.parents.clone_from(&other.parents);
    }
}

#[allow(dead_code)]
impl UnionFind {
    pub fn new(node_count: u32) -> UnionFind {
        UnionFind {
            parents: (0..node_count).collect(),
        }
    }

    /// Removes all nodes
    pub fn clear(&mut self) {
        self.parents.clear();
    }

    /// Does not check the invariants:
    /// - parents[i] <= i
    /// - len <= u32::MAX + 1
    pub fn from_vec_unchecked(parents: Vec<u32>) -> UnionFind {
        UnionFind { parents }
    }

    /// Crates n consecutive new nodes.
    /// Returns first node created.
    pub fn extend(&mut self, n: u32) -> u32 {
        debug_assert!(self.parents.len() as u64 + n as u64 <= (u32::MAX as u64) + 1);
        let first = self.parents.len() as u32;
        self.parents.extend(first..first + n);
        first
    }

    /// Sets the parent for the provided node without checking that:
    /// `parent <= node`
    pub fn set_parent_unchecked(&mut self, node: u32, parent: u32) {
        self.parents[node as usize] = parent;
    }

    /// Finds the root and compresses the path to it by half
    pub fn root_uncached(&self, mut i: u32) -> u32 {
        while i != self.parents[i as usize] {
            let parent = self.parents[i as usize];
            i = self.parents[parent as usize];
        }
        i
    }

    /// Finds the root and compresses the path to it by half
    pub fn root(&mut self, mut i: u32) -> u32 {
        while i != self.parents[i as usize] {
            let parent = self.parents[i as usize];
            let grandparent = self.parents[parent as usize];

            self.parents[i as usize] = grandparent;
            i = grandparent;
        }
        i
    }

    /// Returns the amount of nodes independently of the aliases
    pub fn len(&self) -> u32 {
        self.parents.len() as u32
    }

    pub fn parent(&self, i: u32) -> u32 {
        self.parents[i as usize]
    }

    /// Flattens the node using the grand parent
    pub fn grand_parent(&mut self, node: u32) -> u32 {
        let i = node as usize;
        self.parents[i] = self.parents[self.parents[i] as usize];
        self.parents[i]
    }

    /// Flattens the node checking if it's parent or grand_parent is the requested
    pub fn has_grandparent(&mut self, node: u32, grand_parent: u32) -> bool {
        let parent = self.parent(node);
        if parent == grand_parent {
            return true;
        }
        let found = self.parent(parent);
        if found == grand_parent {
            true
        } else {
            self.parents[node as usize] = found;
            false
        }
    }

    /// Return the root of the resulting alias
    pub fn alias(&mut self, a: u32, b: u32) -> u32 {
        let root_a = self.root(a);
        let root_b = self.root(b);

        if root_a < root_b {
            self.parents[root_b as usize] = root_a;
            root_a
        } else {
            self.parents[root_a as usize] = root_b;
            root_b
        }
    }

    /// Converts the UnionFind alias-datastructure into a dense rename map.
    ///
    /// - The parents of i (vec[i]) will point to what was the root of the node i (root(i)).
    /// - All elements of the returned vec will be in 0..=(the amount of disctinct roots - 1)
    ///   without any value missing.
    ///
    /// Returns (the Vec rename mapping, the amount of disctinct roots).
    pub fn into_compact_rename(mut self) -> (Vec<u32>, u32) {
        let mut root_count = 0;
        for i in 0..self.parents.len() {
            if self.parents[i] == i as u32 {
                self.parents[i] = root_count;
                root_count += 1;
            } else {
                self.parents[i] = self.parents[self.parents[i] as usize];
            }
        }
        (self.parents, root_count)
    }

    /// Resove the backwards dependencies.
    /// After this call, for all i: parent(i) == root(i)
    pub fn flatten(&mut self) {
        for i in 0..self.parents.len() {
            self.grand_parent(i as u32);
        }
    }

    /// Removes all aplied aliases.
    /// Sets the root of every node to itselfs.
    pub fn reset_aliases(&mut self) {
        for i in 0..self.parents.len() {
            self.parents[i] = i as u32;
        }
    }
}
