pub struct DenseMap<V>(Vec<Option<V>>);

#[allow(dead_code)]
impl<V> DenseMap<V> {
    pub fn insert<K: Into<usize>>(&mut self, idx: K, val: V) {
        let idx = idx.into();
        while self.0.len() < idx {
            self.0.push(None);
        }
        self.0.push(Some(val));
    }

    pub fn get<K: TryInto<usize>>(&self, idx: K) -> Option<&V> {
        let idx = idx.try_into().ok()?;
        self.0.get(idx)?.as_ref()
    }

    pub fn get_mut<K: TryInto<usize>>(&mut self, idx: K) -> Option<&mut V> {
        let idx = idx.try_into().ok()?;
        self.0.get_mut(idx)?.as_mut()
    }

    pub fn take<K: TryInto<usize>>(&mut self, idx: K) -> Option<V> {
        let idx = idx.try_into().ok()?;
        let slot: &mut Option<V> = self.0.get_mut(idx)?;
        slot.take()
    }

    pub fn remove<K: TryInto<usize>>(&mut self, idx: K) {
        if let Ok(idx) = idx.try_into()
            && let Some(slot) = self.0.get_mut(idx)
        {
            *slot = None;
        }
    }

    pub fn iter(&self) -> impl Iterator<Item = (usize, &V)> {
        self.0
            .iter()
            .enumerate()
            .flat_map(|(idx, val)| val.as_ref().map(|val| (idx, val)))
    }

    pub fn into_iter(self) -> impl Iterator<Item = (usize, V)> {
        self.0
            .into_iter()
            .enumerate()
            .flat_map(|(idx, val)| val.map(|val| (idx, val)))
    }

    pub fn iter_slots(&mut self) -> impl Iterator<Item = (usize, &mut Option<V>)> {
        self.0.iter_mut().enumerate()
    }
}

impl<V> Default for DenseMap<V> {
    fn default() -> Self {
        Self(Vec::default())
    }
}
