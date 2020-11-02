use std::fmt;
use std::fmt::Formatter;

#[derive(Copy, Clone, Debug)]
pub struct DimensionMismatchError {
    expected_shape: [usize; 2],
    got: [usize; 2]
}

impl DimensionMismatchError {
    pub fn new(expected_shape: [usize; 2], got: [usize; 2]) -> Self
    {
        DimensionMismatchError {
            expected_shape,
            got
        }
    }
}

impl fmt::Display for DimensionMismatchError {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(f, "Expected dimension [{}, {}], Got [{}, {}]", self.expected_shape[0], self.expected_shape[1], self.got[0], self.got[1])
    }
}

#[derive(Copy, Clone, Debug)]
pub struct OutOfBoundError
{
    range: [usize; 2],
    got: [usize; 2]
}

impl OutOfBoundError
{
    pub fn new(range: [usize; 2], got: [usize; 2]) -> Self
    {
        OutOfBoundError
        {
            range,
            got
        }
    }
}

impl fmt::Display for OutOfBoundError
{
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(f, "Max x = {}, Max y = {}, got ({}, {})", self.range[0], self.range[1], self.got[0], self.got[1])
    }
}