package com.deep.framework.jit.statement;

import com.deep.framework.jit.Parser;
import lombok.Data;

import java.util.ArrayList;
import java.util.List;
import java.util.Objects;
import java.util.stream.Collectors;

import static com.deep.framework.jit.lexer.TokenType.SEMI;

@Data
public class PackageStatement implements Statement {
    public List<Object> packages = new ArrayList<>();

    public void parser(Statement parent, Object obj, List<Object> list) {
        Parser.ends.add(SEMI);
        Parser.statements.add(this);
        while (true) {
            Object o = list.remove(0);
            if (o.equals(Parser.ends.get(0))) {
                Parser.ends.remove(0);
                return;
            } else {
                packages.add(o);
            }
        }
    }

    @Override
    public String toString() {
        return packages.stream()
           .map(Objects::toString)
           .collect(Collectors.joining());
    }
}
