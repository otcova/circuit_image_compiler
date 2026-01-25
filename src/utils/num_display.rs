use std::fmt;

pub struct SiValue<N>(pub N);

impl<N: Into<f64> + Copy> fmt::Display for SiValue<N> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        const SIG_FIGS: i32 = 3;

        let value: f64 = self.0.into();

        if value == f64::INFINITY {
            return write!(f, "Infinity");
        } else if value == f64::NEG_INFINITY {
            return write!(f, "-Infinity");
        } else if value.is_nan() {
            return write!(f, "NaN");
        }

        let prefixes = [(1e12, "T"), (1e9, "G"), (1e6, "M"), (1e3, "k")];

        let abs = value.abs();
        let (scaled, suffix) = prefixes
            .iter()
            .find(|(factor, _)| abs >= *factor)
            .map(|(factor, suffix)| (value / factor, *suffix))
            .unwrap_or((value, ""));

        // Number of digits before the decimal point
        let int_digits = scaled.abs().log10().floor() as i32 + 1;

        // Precision needed to reach SIG_FIGS
        let precision = (SIG_FIGS - int_digits).max(0) as usize;

        let pow = 10f64.powi(precision as i32);
        let rounded = (scaled * pow).round() / pow;

        write!(f, "{}{}", rounded, suffix)

        // Alternative with fixed decimals:
        // write!(f, "{:.*}{}", precision, scaled, suffix)
    }
}

pub struct GroupedUInt<N>(pub N);

impl<N: Into<u64> + Copy> fmt::Display for GroupedUInt<N> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let val: u64 = self.0.into();

        if val == 0 {
            return write!(f, "0");
        }

        // To avoid allocations, we determine the largest power of 1000
        // that is less than or equal to our value.
        let mut divisor = 1;
        while val / (divisor * 1000) > 0 {
            divisor *= 1000;
        }

        // Write the first group (could be 1, 2, or 3 digits)
        let first_group = val / divisor;
        write!(f, "{}", first_group)?;

        // Move to the next groups, which must always be 3 digits (padded with 0s)
        while divisor > 1 {
            divisor /= 1000;
            let group = (val / divisor) % 1000;
            // The {:03} ensures groups like 1_005 are not printed as 1_5
            write!(f, ",{:03}", group)?;
        }

        Ok(())
    }
}
