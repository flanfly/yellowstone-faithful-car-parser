use {
    crate::{util, varint},
    cid::Cid,
    crc::{CRC_64_GO_ISO, Crc},
    fnv::FnvHasher,
    indexmap::IndexMap,
    std::{fmt, hash::Hasher, io},
    thiserror::Error,
    tokio::io::AsyncRead,
};
pub use {
    block::{Block, Shredding, SlotMeta},
    dataframe::DataFrame,
    entry::Entry,
    epoch::Epoch,
    rewards::Rewards,
    subset::Subset,
    transaction::Transaction,
};

mod block;
mod dataframe;
mod entry;
mod epoch;
mod rewards;
mod subset;
mod transaction;

const MAX_ALLOWED_SECTION_SIZE: usize = 32 << 20; // 32MiB

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Kind {
    Transaction,
    Entry,
    Block,
    Subset,
    Epoch,
    Rewards,
    DataFrame,
}

impl Kind {
    pub const fn from_u64(kind: u64) -> Option<Kind> {
        match kind {
            0 => Some(Kind::Transaction),
            1 => Some(Kind::Entry),
            2 => Some(Kind::Block),
            3 => Some(Kind::Subset),
            4 => Some(Kind::Epoch),
            5 => Some(Kind::Rewards),
            6 => Some(Kind::DataFrame),
            _ => None,
        }
    }

    pub const fn to_u64(&self) -> u64 {
        match self {
            Kind::Transaction => 0,
            Kind::Entry => 1,
            Kind::Block => 2,
            Kind::Subset => 3,
            Kind::Epoch => 4,
            Kind::Rewards => 5,
            Kind::DataFrame => 6,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum Node {
    Transaction(transaction::Transaction),
    Entry(entry::Entry),
    Block(block::Block),
    Subset(subset::Subset),
    Epoch(epoch::Epoch),
    Rewards(rewards::Rewards),
    DataFrame(dataframe::DataFrame),
}

impl TryFrom<&[u8]> for Node {
    type Error = NodeError;

    fn try_from(value: &[u8]) -> Result<Self, Self::Error> {
        let value = serde_cbor::from_slice::<serde_cbor::Value>(value)?;

        let mut kind = None;
        if let serde_cbor::Value::Array(vec) = &value {
            if let Some(serde_cbor::Value::Integer(value)) = vec.first() {
                kind = Some(*value as u64);
            }
        }
        let Some(kind) = kind.and_then(Kind::from_u64) else {
            return Err(NodeError::UnknownKind(kind));
        };

        Ok(match kind {
            Kind::Transaction => Node::Transaction(Transaction::try_from(value)?),
            Kind::Entry => Node::Entry(Entry::try_from(value)?),
            Kind::Block => Node::Block(Block::try_from(value)?),
            Kind::Subset => Node::Subset(Subset::try_from(value)?),
            Kind::Epoch => Node::Epoch(Epoch::try_from(value)?),
            Kind::Rewards => Node::Rewards(Rewards::try_from(value)?),
            Kind::DataFrame => Node::DataFrame(DataFrame::try_from(value)?),
        })
    }
}

impl Node {
    pub const fn kind(&self) -> Kind {
        match self {
            Self::Transaction(_) => Kind::Transaction,
            Self::Entry(_) => Kind::Entry,
            Self::Block(_) => Kind::Block,
            Self::Subset(_) => Kind::Subset,
            Self::Epoch(_) => Kind::Epoch,
            Self::Rewards(_) => Kind::Rewards,
            Self::DataFrame(_) => Kind::DataFrame,
        }
    }
}

#[derive(Debug, Clone)]
pub struct NodeWithCid {
    pub cid: Cid,
    pub node: Node,
}

impl TryFrom<&RawNode> for NodeWithCid {
    type Error = NodeError;

    fn try_from(value: &RawNode) -> Result<Self, Self::Error> {
        Node::try_from(value.get_data()).map(|node| Self {
            cid: value.cid,
            node,
        })
    }
}

#[derive(Debug, Clone)]
pub struct RawNode {
    pub cid: Cid,
    data: Vec<u8>,
    data_offset: usize,
}

impl RawNode {
    pub const fn new(cid: Cid, data: Vec<u8>) -> RawNode {
        RawNode {
            cid,
            data,
            data_offset: 0,
        }
    }

    pub fn new_from_vec(data: Vec<u8>) -> Result<Self, NodeError> {
        let mut buf = data.as_slice();

        let cid_version = varint::decode_varint(&mut buf)?;
        if !matches!(cid_version, 0 | 1) {
            return Err(NodeError::UnknownCid(cid_version));
        }

        let multicodec = varint::decode_varint(&mut buf)?;

        let hash_function = varint::decode_varint(&mut buf)?;
        let digest_length = varint::decode_varint(&mut buf)? as usize;
        if buf.len() < digest_length {
            return Err(NodeError::MultihashNotEnoughBytes);
        }
        let ha = multihash::Multihash::wrap(hash_function, &buf[0..digest_length])?;

        // calculate data offset
        let data_offset = data.len() - buf.len() + digest_length;

        let cid = match cid_version {
            0 => Cid::new_v0(ha)?,
            1 => Cid::new_v1(multicodec, ha),
            _ => unreachable!(),
        };

        Ok(RawNode {
            cid,
            data,
            data_offset,
        })
    }

    pub fn get_data(&self) -> &[u8] {
        &self.data[self.data_offset..]
    }
}

pub struct NodeReader<R> {
    reader: R,
    header: Vec<u8>,
}

impl<R> fmt::Debug for NodeReader<R> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("NodeReader").finish()
    }
}

impl<R: AsyncRead + Unpin> NodeReader<R> {
    pub const fn new(reader: R) -> Self {
        Self {
            reader,
            header: vec![],
        }
    }

    pub async fn read_header(&mut self) -> Result<&[u8], NodeError> {
        if self.header.is_empty() {
            let header_length = varint::read(&mut self.reader).await? as usize;
            self.header = util::read_exact(&mut self.reader, header_length).await?;
        }

        Ok(&self.header)
    }

    pub async fn read_node(&mut self) -> Result<Option<RawNode>, NodeError> {
        if self.header.is_empty() {
            self.read_header().await?;
        };

        // read and decode the uvarint prefix (length of CID + data)
        let section_size = match varint::read(&mut self.reader).await {
            Ok(size) => size as usize,
            Err(varint::VarIntError::Io(error)) if error.kind() == io::ErrorKind::UnexpectedEof => {
                return Ok(None);
            }
            Err(error) => return Err(error.into()),
        };
        if section_size > MAX_ALLOWED_SECTION_SIZE {
            return Err(NodeError::SectionTooLong(section_size));
        }

        let section = util::read_exact(&mut self.reader, section_size).await?;
        RawNode::new_from_vec(section).map(Some)
    }
}

#[derive(Debug, Default, Clone)]
pub struct Nodes {
    pub nodes: IndexMap<Cid, Node>,
}

impl Nodes {
    pub async fn read_until_block<R: AsyncRead + Unpin>(
        reader: &mut NodeReader<R>,
    ) -> Result<Self, NodeError> {
        let mut block = Self::default();
        let mut finished = false;
        while !finished {
            let Some(node) = reader.read_node().await? else {
                break;
            };
            let node = NodeWithCid::try_from(&node)?;
            finished = matches!(node.node, Node::Block(_));
            block.push(node);
        }
        Ok(block)
    }

    pub fn push(&mut self, node: NodeWithCid) {
        self.nodes.insert(node.cid, node.node);
    }

    pub fn reassemble_dataframes<'a>(
        &'a self,
        dataframe: &'a DataFrame,
    ) -> Result<Vec<u8>, ReassableError> {
        let mut dataframes = vec![dataframe];
        let total = dataframe.total.unwrap_or_default() as usize;
        if total > 1 {
            let mut next_cids = dataframe.next.clone();
            while let Some(cid) = next_cids.pop() {
                let Some(node) = self.nodes.get(&cid) else {
                    return Err(ReassableError::MissedCid(cid));
                };
                let Node::DataFrame(dataframe) = node else {
                    return Err(ReassableError::InvalidNode(node.kind()));
                };
                next_cids.extend(&dataframe.next);
                dataframes.push(dataframe);
            }
            if dataframes.len() != total {
                return Err(ReassableError::InvalidNumberOfDataFrames {
                    expected: total,
                    found: dataframes.len(),
                });
            }
            dataframes.sort_by_key(|frame| frame.index.unwrap_or_default());
        }

        let data: Vec<u8> = dataframes
            .iter()
            .flat_map(|frame| frame.data.iter())
            .copied()
            .collect();

        if let Some(expected) = dataframe.hash {
            let crc64 = get_crc64(&data);
            if crc64 != expected {
                // maybe it's the legacy checksum function?
                let fnv = get_fnv(&data);
                if fnv != expected {
                    return Err(ReassableError::InvalidHash {
                        crc64,
                        fnv,
                        expected,
                    });
                }
            }
        }

        Ok(data)
    }
}

fn get_crc64(data: &[u8]) -> u64 {
    let crc = Crc::<u64>::new(&CRC_64_GO_ISO);
    let mut digest = crc.digest();
    digest.update(data);
    digest.finalize()
}

fn get_fnv(data: &[u8]) -> u64 {
    let mut hasher = FnvHasher::default();
    hasher.write(data);
    hasher.finish()
}

#[derive(Debug, Error)]
pub enum NodeError {
    // read
    #[error(transparent)]
    Io(#[from] io::Error),
    #[error("invalid varint")]
    InvalidVarInt,
    #[error("section size is too long: {0} (max {max})", max = MAX_ALLOWED_SECTION_SIZE)]
    SectionTooLong(usize),
    // deserialize
    #[error(transparent)]
    DeserializeCbor(#[from] serde_cbor::Error),
    #[error("expected {kind:?} for {path:?}")]
    UnexpectedCborValue {
        path: &'static str,
        kind: &'static str,
    },
    #[error("too much items in the array")]
    UnexpectedCborValues,
    #[error("invalid node kind: {node:?} (expected: {expected:?})")]
    InvalidKind { node: u64, expected: u64 },
    #[error("unknown node kind: {0:?}")]
    UnknownKind(Option<u64>),
    #[error(transparent)]
    InvalidCid(#[from] cid::Error),
    #[error("unknown cid version: {0} (expected 0 or 1)")]
    UnknownCid(u64),
    #[error("not enough bytes for multihash")]
    MultihashNotEnoughBytes,
    #[error(transparent)]
    InvalidMultihash(#[from] multihash::Error),
}

impl From<varint::VarIntError> for NodeError {
    fn from(value: varint::VarIntError) -> Self {
        match value {
            varint::VarIntError::Io(error) => Self::Io(error),
            varint::VarIntError::Invalid => Self::InvalidVarInt,
        }
    }
}

impl NodeError {
    #[inline]
    pub const fn assert_invalid_kind(node: u64, expected: Kind) -> Result<(), Self> {
        if node == expected.to_u64() {
            Ok(())
        } else {
            Err(Self::InvalidKind {
                node,
                expected: expected.to_u64(),
            })
        }
    }
}

#[derive(Debug, Error)]
pub enum ReassableError {
    #[error("missed cid: {0}")]
    MissedCid(Cid),
    #[error("expected {expected} DataFrames, found: {found}")]
    InvalidNumberOfDataFrames { expected: usize, found: usize },
    #[error("invalid node kind: {0:?} (expected DataFrame)")]
    InvalidNode(Kind),
    #[error("invalid hash: crc64/{crc64} fnv/{fnv} (expected: {expected}")]
    InvalidHash { crc64: u64, fnv: u64, expected: u64 },
}
